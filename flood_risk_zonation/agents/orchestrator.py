"""
PRAVAAH — Agent Orchestrator.

The orchestrator controls the agentic decision-support workflow.  It:
  1. Receives a habitation ID and the full FullSIHResult context.
  2. Determines which agents to invoke (based on priority class).
  3. Calls tools to retrieve structured PRAVAAH data.
  4. Invokes agents in the correct sequence.
  5. Aggregates evidence into a final AgentDecision.
  6. Returns the AgentDecision — never a raw LLM response.

BOUNDED EXECUTION:
  - Each habitation analysis involves at most 5 agent calls.
  - LLM is only invoked for HIGH and CRITICAL habitations.
  - LLM is NOT invoked for LOW-risk habitations (purely rule-based).
  - Maximum execution time is enforced by each agent's LLM timeout.
  - Orchestrator never loops back to agents after a decision is reached.

DETERMINISTIC FALLBACK:
  If any agent's LLM call fails, the agent returns a rule-based fallback.
  The orchestrator assembles a valid AgentDecision from whichever outputs
  are available.  The AgentDecision.ai_assisted field reflects whether any
  LLM contributed.  The UI displays:
    "AI explanation unavailable — displaying rule-based decision."
  when ai_assisted is False.

WORKFLOW:
  For LOW priority:
    → Hazard agent only (deterministic fallback; no LLM)

  For MEDIUM priority:
    → Hazard agent + Exposure agent (deterministic fallback)

  For HIGH / CRITICAL priority:
    → Full workflow: Hazard → Exposure → Vulnerability → Capacity
                   → find_candidates → compare_candidates
                   → Relocation Planner
"""
from __future__ import annotations

import logging
import time
from typing import Any

from flood_risk_zonation.agents.agents import (
    run_hazard_agent,
    run_exposure_agent,
    run_vulnerability_agent,
    run_capacity_agent,
    run_relocation_agent,
    _llm_available,
)
from flood_risk_zonation.agents.tools import (
    get_hazard_details,
    get_exposure_details,
    get_vulnerability_details,
    get_capacity_details,
    get_relocation_details,
    find_relocation_candidates_tool,
    compare_relocation_candidates_tool,
)
from flood_risk_zonation.models import AgentDecision, AgentEvidence

logger = logging.getLogger(__name__)


class PravaahOrchestrator:
    """
    Bounded agentic orchestrator for PRAVAAH decision support.

    Parameters
    ----------
    full_result : FullSIHResult
        Complete Phase 3 result providing access to all pipeline data.
    verbose : bool
        If True, log each agent call and its result.
    """

    def __init__(self, full_result: Any, verbose: bool = False) -> None:
        self._result = full_result
        self._verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyse_habitation(self, hab_id: str) -> AgentDecision:
        """
        Run the full decision-support workflow for one habitation.

        Parameters
        ----------
        hab_id : str
            The habitation to analyse.

        Returns
        -------
        AgentDecision
            Fully structured decision with evidence, candidates, and recommendation.
        """
        t0 = time.time()
        sih = self._result.sih_result

        # ── Look up relocation priority (determines workflow depth) ───────────
        rel = next((r for r in sih.relocation_results if r.hab_id == hab_id), None)
        if rel is None:
            return self._no_data_decision(hab_id, "No relocation priority data found.")

        priority = rel.priority_class
        zone = self._result.habitation_zones.get(hab_id, "UNKNOWN")

        if self._verbose:
            logger.info("Orchestrator: %s → priority=%s zone=%s", hab_id, priority, zone)

        # ── Retrieve tool data ────────────────────────────────────────────────
        try:
            hazard_data = get_hazard_details(
                hab_id,
                sih.exposure_results,
                self._result.zoned_grid,
                self._result.habitation_zones,
                sih.flood_risk_result.scored_grid,
            )
        except ValueError as e:
            return self._no_data_decision(hab_id, str(e))

        exposure_data = None
        vuln_data     = None
        cap_data      = None
        rel_data      = None
        candidates    = []
        comparison    = {"best_candidate_id": None, "comparison_narrative": "No candidates found.", "ranking": []}

        try:
            exposure_data = get_exposure_details(hab_id, sih.exposure_results)
        except ValueError:
            pass

        if priority in ("MEDIUM", "HIGH", "CRITICAL"):
            try:
                vuln_data = get_vulnerability_details(hab_id, sih.vulnerability_results)
            except ValueError:
                pass

        if priority in ("HIGH", "CRITICAL"):
            try:
                cap_data  = get_capacity_details(hab_id, sih.capacity_results)
                rel_data  = get_relocation_details(hab_id, sih.relocation_results)
                candidates = find_relocation_candidates_tool(hab_id, self._result.relocation_candidates)
                comparison = compare_relocation_candidates_tool(candidates)
            except ValueError:
                pass

        # ── Invoke agents ─────────────────────────────────────────────────────
        evidence: list[AgentEvidence] = []

        # Hazard agent — always runs
        h_ev = run_hazard_agent(hazard_data)
        evidence.append(h_ev)

        # Exposure agent — runs for MEDIUM+
        if exposure_data and priority in ("MEDIUM", "HIGH", "CRITICAL"):
            e_ev = run_exposure_agent(exposure_data)
            evidence.append(e_ev)

        # Vulnerability agent — runs for HIGH+
        if vuln_data and priority in ("HIGH", "CRITICAL"):
            v_ev = run_vulnerability_agent(vuln_data)
            evidence.append(v_ev)

        # Capacity agent — runs for HIGH+
        if cap_data and priority in ("HIGH", "CRITICAL"):
            c_ev = run_capacity_agent(cap_data)
            evidence.append(c_ev)

        # Relocation planner — runs for HIGH+ with all evidence
        reloc_ev = None
        if rel_data and priority in ("HIGH", "CRITICAL"):
            reloc_ev = run_relocation_agent(rel_data, evidence, candidates, comparison)
            evidence.append(reloc_ev)

        # ── Assemble final AgentDecision ──────────────────────────────────────
        any_ai = any(e.ai_assisted for e in evidence)
        fallback_reason = "" if any_ai else (
            "LLM unavailable — rule-based decision"
            if not _llm_available()
            else "LLM not invoked for this priority level"
        )

        # Build top-level summary from the most relevant agent
        if reloc_ev:
            top_summary = reloc_ev.summary
            action      = rel_data["recommended_action"] if rel_data else rel.recommended_action
        elif evidence:
            top_summary = evidence[-1].summary
            action      = rel.recommended_action
        else:
            top_summary = f"Priority: {priority}. No detailed analysis available."
            action      = rel.recommended_action

        # Candidate objects (not just dicts) for the decision model
        candidate_objects = self._result.relocation_candidates.get(hab_id, [])

        # Top candidate reason from comparison
        top_reason = ""
        if candidate_objects:
            best_id = comparison.get("best_candidate_id")
            best_cand = next((c for c in candidate_objects if c.candidate_id == best_id), None)
            if best_cand:
                top_reason = best_cand.notes

        duration = time.time() - t0
        if self._verbose:
            logger.info("Orchestrator: %s done in %.2fs (ai=%s)", hab_id, duration, any_ai)

        return AgentDecision(
            hab_id=hab_id,
            hab_name=rel.name or "Unnamed",
            priority_class=priority,
            relocation_score=rel.relocation_score,
            spatial_zone=zone,
            summary=top_summary,
            recommended_action=action,
            evidence=evidence,
            candidate_areas=candidate_objects,
            top_candidate_reason=top_reason,
            ai_assisted=any_ai,
            fallback_reason=fallback_reason,
        )

    def analyse_all(
        self,
        priority_filter: tuple[str, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW"),
        max_habitations: int = 50,
    ) -> dict[str, AgentDecision]:
        """
        Run decision-support analysis for all (or filtered) habitations.

        Parameters
        ----------
        priority_filter : tuple[str, ...]
            Only analyse habitations whose relocation priority is in this tuple.
        max_habitations : int
            Hard cap to prevent excessive LLM costs.

        Returns
        -------
        dict[str, AgentDecision]  — hab_id → AgentDecision
        """
        sih = self._result.sih_result
        decisions: dict[str, AgentDecision] = {}

        # Sort by priority (CRITICAL first) to process most important first
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        candidates_list = [
            r for r in sih.relocation_results
            if r.priority_class in priority_filter
        ]
        candidates_list = sorted(
            candidates_list,
            key=lambda r: (order.get(r.priority_class, 9), -r.relocation_score),
        )[:max_habitations]

        for rel in candidates_list:
            try:
                decision = self.analyse_habitation(rel.hab_id)
                decisions[rel.hab_id] = decision
            except Exception as exc:
                logger.warning("Orchestrator: analysis failed for %s: %s", rel.hab_id, exc)

        logger.info(
            "Orchestrator: analysed %d habitations (ai=%d, rule-based=%d)",
            len(decisions),
            sum(1 for d in decisions.values() if d.ai_assisted),
            sum(1 for d in decisions.values() if not d.ai_assisted),
        )
        return decisions

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _no_data_decision(self, hab_id: str, reason: str) -> AgentDecision:
        """Return a minimal AgentDecision when data is unavailable."""
        return AgentDecision(
            hab_id=hab_id,
            hab_name=hab_id,
            priority_class="UNKNOWN",
            relocation_score=0.0,
            spatial_zone="UNKNOWN",
            summary=f"Decision-support analysis unavailable: {reason}",
            recommended_action="Manual review required — insufficient data for automated assessment.",
            evidence=[],
            candidate_areas=[],
            ai_assisted=False,
            fallback_reason=reason,
        )
