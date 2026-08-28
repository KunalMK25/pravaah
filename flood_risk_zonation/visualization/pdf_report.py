"""PRAVAAH-AI — PDF Report Generator.

Produces a structured risk-assessment report covering hazard analysis,
vulnerable habitation exposure, vulnerability scores, carrying-capacity
status, and relocation priority recommendations.
"""
from __future__ import annotations
import io
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

RISK_COLORS = {"High": colors.HexColor("#e74c3c"), "Medium": colors.HexColor("#f39c12"),
               "Low": colors.HexColor("#2ecc71"), "Water": colors.HexColor("#3498db")}
RISK_BG = {"High": colors.HexColor("#fdecea"), "Medium": colors.HexColor("#fef9e7"),
           "Low": colors.HexColor("#eafaf1"), "Water": colors.HexColor("#ebf5fb")}

def _make_styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("Title2", parent=s["Title"], fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a252f")))
    s.add(ParagraphStyle("Subtitle", parent=s["Normal"], fontSize=11, textColor=colors.HexColor("#5d6d7e"), spaceAfter=12))
    s.add(ParagraphStyle("SectionHead", parent=s["Heading1"], fontSize=13, textColor=colors.HexColor("#1a5276"),
                         spaceBefore=14, spaceAfter=6, borderPad=4))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=9.5, leading=14, spaceAfter=6, alignment=TA_JUSTIFY))
    s.add(ParagraphStyle("BulletBody", parent=s["Normal"], fontSize=9.5, leading=14, leftIndent=12, spaceAfter=4))
    s.add(ParagraphStyle("Caption", parent=s["Normal"], fontSize=8, textColor=colors.grey, alignment=TA_CENTER))
    s.add(ParagraphStyle("TableHeader", parent=s["Normal"], fontSize=9, textColor=colors.white,
                         alignment=TA_CENTER, fontName="Helvetica-Bold"))
    return s

def _risk_distribution_chart(dist: dict) -> io.BytesIO:
    labels = [k for k in ["High", "Medium", "Low", "Water"] if k in dist]
    values = [dist[k] for k in labels]
    clrs = [{"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71", "Water": "#3498db"}[k] for k in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.bar(labels, values, color=clrs, edgecolor="white", linewidth=0.8)
    ax1.set_ylabel("Number of Grid Cells", fontsize=9)
    ax1.set_title("Risk Class Distribution", fontsize=10, fontweight="bold")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)
    non_water = {k: v for k, v in dist.items() if k != "Water"}
    if non_water:
        pie_labels = list(non_water.keys())
        pie_vals = list(non_water.values())
        pie_clrs = [{"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}[k] for k in pie_labels]
        ax2.pie(pie_vals, labels=pie_labels, colors=pie_clrs, autopct="%1.0f%%",
                startangle=90, textprops={"fontsize": 9})
        ax2.set_title("Risk Share (excl. Water)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return buf

def _feature_importance_chart(importances: dict) -> io.BytesIO:
    items = sorted(importances.items(), key=lambda x: x[1])[-8:]
    labels = [k.replace("_", " ").title() for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(labels, vals, color="#3498db", edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title("Flood Susceptibility Factor Weights", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return buf

def _vulnerability_analysis(result, area_name: str) -> list[str]:
    grid = result.scored_grid
    dist = result.risk_distribution
    total = sum(v for k, v in dist.items() if k != "Water")
    high_pct = dist.get("High", 0) / total * 100 if total else 0
    med_pct = dist.get("Medium", 0) / total * 100 if total else 0
    reasons = []
    if "elevation_m" in grid.columns:
        elev_min = grid[grid.risk_class == "High"]["elevation_m"].mean() if dist.get("High", 0) > 0 else 0
        elev_all = grid["elevation_m"].mean()
        if elev_min < elev_all - 5:
            reasons.append(f"Low-lying terrain: High-risk cells average {elev_min:.0f}m elevation vs area mean {elev_all:.0f}m — depressions where water accumulates.")
    if "drainage_capacity" in grid.columns:
        drain_high = grid[grid.risk_class == "High"]["drainage_capacity"].mean() if dist.get("High", 0) > 0 else 0.5
        if drain_high < 0.45:
            reasons.append(f"Low drainage infrastructure proxy score: High-risk cells have an average OSM-derived drainage proxy score of {drain_high:.2f}/1.0 (heuristic; not measured municipal capacity) — low score indicates sparse mapped drainage linestrings nearby.")
    if "dist_water_m" in grid.columns:
        dist_high = grid[grid.risk_class == "High"]["dist_water_m"].mean() if dist.get("High", 0) > 0 else 5000
        if dist_high < 1000:
            reasons.append(f"Proximity to water bodies: High-risk cells are on average {dist_high:.0f}m from the nearest lake or canal — within overflow range during heavy rain.")
    if "twi" in grid.columns:
        twi_high = grid[grid.risk_class == "High"]["twi"].mean() if dist.get("High", 0) > 0 else 0
        twi_all = grid["twi"].mean()
        if twi_high > twi_all * 1.2:
            reasons.append(f"High Topographic Wetness Index (TWI): High-risk cells average TWI of {twi_high:.2f} vs area mean {twi_all:.2f} — terrain naturally funnels water to these zones.")
    if "rainfall_mean_mm" in grid.columns:
        rain = grid["rainfall_mean_mm"].mean()
        reasons.append(f"Rainfall intensity: Mean annual rainfall of {rain:.0f}mm, concentrated in June–September monsoon season. Extreme events (100mm+ in 24h) are the primary flood trigger.")
    if dist.get("Water", 0) > 0:
        reasons.append(f"{dist['Water']} permanent water bodies (lakes, tanks, canals) identified — surrounding cells are at elevated risk from overflow and seepage.")
    return reasons

def _emergency_plan(dist: dict, grid) -> list[tuple]:
    total = sum(v for k, v in dist.items() if k != "Water")
    high = dist.get("High", 0); med = dist.get("Medium", 0); low = dist.get("Low", 0)
    plan = [
        ("🔴 High Risk Zones", f"{high} cells ({high/total*100:.0f}%)",
         "• Deploy rescue boats and swift-water rescue teams\n• Pre-position emergency shelters within 500m\n• Establish medical first-response units\n• Mandatory evacuation orders for residents\n• 24/7 monitoring with water level sensors\n• Food & water supply for 72-hour self-sufficiency"),
        ("🟡 Medium Risk Zones", f"{med} cells ({med/total*100:.0f}%)",
         "• Standby rescue teams on 30-minute alert\n• Voluntary evacuation advisory for vulnerable groups\n• Pre-stock sandbags and pumping equipment\n• Open community shelters as precautionary measure\n• Regular patrol every 4 hours during heavy rain\n• Food distribution points at zone boundaries"),
        ("🟢 Low Risk Zones", f"{low} cells ({low/total*100:.0f}%)",
         "• Normal monitoring — daily check-ins\n• Use as staging area for rescue operations\n• Set up relief camps and food distribution hubs\n• Medical camps for evacuees from High/Medium zones\n• Coordinate volunteer registration and deployment\n• Maintain communication infrastructure"),
        ("🔵 Water Bodies", f"{dist.get('Water', 0)} cells",
         "• Deploy boats for water rescue operations\n• Monitor water levels every hour\n• Coordinate with BBMP lake management authority\n• Alert downstream communities if overflow detected\n• No civilian access — restrict to rescue personnel"),
    ]
    return plan

def _risk_zone_parameters_table(result, styles) -> list:
    """Build a table explaining risk zone thresholds and feature parameters."""
    config = result.config
    grid = result.scored_grid
    story = []

    story.append(Paragraph("Risk Zone Classification Parameters", styles["SectionHead"]))

    # Threshold table
    thresh_data = [
        ["Risk Zone", "Score Range", "Color", "Elevation (typical)", "Drainage", "Dist. to Water"],
        ["🔵 Water", "N/A (masked)", "Blue", "≤ 1m OR inside OSM water polygon", "N/A", "0m"],
        ["🔴 High Risk", f"> {config.medium_threshold:.0f}", "Red",
         f"< area mean by 5m+", "< 0.45 capacity", "< 500m"],
        ["🟡 Medium Risk", f"{config.low_threshold:.0f} – {config.medium_threshold:.0f}", "Amber",
         "Near area mean", "0.45 – 0.65", "500m – 2000m"],
        ["🟢 Low Risk", f"≤ {config.low_threshold:.0f}", "Green",
         "> area mean", "> 0.65 capacity", "> 2000m"],
    ]
    tt = Table(thresh_data, colWidths=[3*cm, 2.5*cm, 1.5*cm, 4*cm, 2.5*cm, 3*cm])
    tt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#dee2e6")),
        ("PADDING", (0,0), (-1,-1), 5),
        ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#ebf5fb")),
        ("BACKGROUND", (0,2), (-1,2), colors.HexColor("#fdecea")),
        ("BACKGROUND", (0,3), (-1,3), colors.HexColor("#fef9e7")),
        ("BACKGROUND", (0,4), (-1,4), colors.HexColor("#eafaf1")),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ]))
    story.append(tt)
    story.append(Spacer(1, 0.3*cm))

    # Declared factor weights used by the transparent susceptibility index.
    story.append(Paragraph("Declared Susceptibility Factor Weights", styles["SectionHead"]))
    fi = result.analysis_result.feature_importances
    fi_interp = {
        "elevation_m": "Lower = higher risk", "twi": "Higher = more water accumulation",
        "drainage_capacity": "Lower = worse drainage", "dist_water_m": "Closer = higher risk",
        "rainfall_mean_mm": "Higher = more water input", "slope_deg": "Flatter = water pools",
        "rainfall_max_24h_mm": "Higher = flash flood risk", "population_density": "Higher = less absorption",
        "aspect_deg": "Affects runoff direction", "curvature": "Concave = water collects",
    }
    fw_data = [["#", "Feature", "Weight (%)", "Risk Direction", "This Area Value (avg)"]]
    for rank, (feat, imp) in enumerate(sorted(fi.items(), key=lambda x: x[1], reverse=True), 1):
        avg_val = ""
        if feat in grid.columns:
            non_water = grid[grid.risk_class != "Water"]
            if len(non_water) > 0:
                avg_val = f"{non_water[feat].mean():.2f}"
        fw_data.append([
            str(rank),
            feat.replace("_", " ").title(),
            f"{imp*100:.1f}%",
            fi_interp.get(feat, "—"),
            avg_val,
        ])
    fw_t = Table(fw_data, colWidths=[1*cm, 4*cm, 2.5*cm, 5*cm, 4*cm])
    fw_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#dee2e6")),
        ("PADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("ALIGN", (0,0), (0,-1), "CENTER"),
        ("ALIGN", (2,0), (2,-1), "CENTER"),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ]))
    story.append(fw_t)
    story.append(Spacer(1, 0.3*cm))

    # Area statistics comparison
    story.append(Paragraph("Area Statistics by Risk Zone", styles["SectionHead"]))
    stat_cols = ["elevation_m", "twi", "drainage_capacity", "dist_water_m", "rainfall_mean_mm"]
    stat_names = ["Elevation (m)", "TWI", "Drainage Cap.", "Dist. Water (m)", "Rainfall (mm)"]
    stat_data = [["Risk Zone"] + stat_names]
    for rc in ["High", "Medium", "Low"]:
        zone = grid[grid.risk_class == rc]
        if len(zone) == 0:
            continue
        row = [rc]
        for col in stat_cols:
            if col in zone.columns:
                row.append(f"{zone[col].mean():.1f}")
            else:
                row.append("—")
        stat_data.append(row)
    # Add area average
    non_water = grid[grid.risk_class != "Water"]
    avg_row = ["Area Average"]
    for col in stat_cols:
        if col in non_water.columns:
            avg_row.append(f"{non_water[col].mean():.1f}")
        else:
            avg_row.append("—")
    stat_data.append(avg_row)

    st_t = Table(stat_data, colWidths=[3*cm, 2.5*cm, 2*cm, 2.5*cm, 3*cm, 3.5*cm])
    st_style = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#dee2e6")),
        ("PADDING", (0,0), (-1,-1), 5),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ]
    for i, row in enumerate(stat_data[1:], 1):
        rc = row[0]
        if rc in RISK_BG:
            st_style.append(("BACKGROUND", (0,i), (-1,i), RISK_BG[rc]))
        elif rc == "Area Average":
            st_style.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#f0f3f4")))
            st_style.append(("FONTNAME", (0,i), (-1,i), "Helvetica-Bold"))
    st_t.setStyle(TableStyle(st_style))
    story.append(st_t)
    return story


def _sih_executive_summary(sih_result, area_name: str, styles) -> list:
    """Build the PRAVAAH-AI executive summary section."""
    story = []
    story.append(Paragraph("PRAVAAH-AI — Executive Summary", styles["Title2"]))
    story.append(Paragraph(
        "Predictive Risk &amp; Vulnerability Assessment for At-Risk Habitations (PRAVAAH-AI)",
        styles["Subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#c0392b"), spaceAfter=10))

    n_habs = len(sih_result.habitation_dataset.habitations)
    n_red = len(sih_result.red_zone_habitations)
    n_critical = len(sih_result.critical_habitations)
    n_high = sum(1 for r in sih_result.relocation_results if r.priority_class == "HIGH")
    n_medium = sum(1 for r in sih_result.relocation_results if r.priority_class == "MEDIUM")
    n_low = sum(1 for r in sih_result.relocation_results if r.priority_class == "LOW")

    summary_data = [
        ["SIH Metric", "Value", "Status"],
        ["Total Habitations Identified", str(n_habs), "—"],
        ["Habitations in Red Zone (High Risk)", str(n_red),
         "⚠️ Action Required" if n_red > 0 else "✅ None"],
        ["CRITICAL Relocation Priority", str(n_critical),
         "🚨 Immediate Action" if n_critical > 0 else "✅ None"],
        ["HIGH Priority (Evacuation Planning)", str(n_high),
         "⚠️ Priority Review" if n_high > 0 else "✅ None"],
        ["MEDIUM Priority (Preparedness)", str(n_medium), "—"],
        ["LOW Priority (Monitoring)", str(n_low), "—"],
        ["Habitation Data Source", sih_result.habitation_dataset.source, "—"],
        ["SIH Analysis Duration", f"{sih_result.sih_duration_seconds:.1f}s", "—"],
    ]
    t = Table(summary_data, colWidths=[7*cm, 4*cm, 6*cm])
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c0392b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
    ]
    if n_critical > 0:
        for i, row in enumerate(summary_data[1:], 1):
            if "CRITICAL" in row[0]:
                t_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fdecea")))
                t_style.append(("TEXTCOLOR", (0, i), (0, i), colors.HexColor("#c0392b")))
                t_style.append(("FONTNAME", (0, i), (-1, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(t_style))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    # Key decision message
    if n_critical > 0:
        msg = (
            f"<b>⚠️ ALERT:</b> {n_critical} habitation(s) have been assessed as CRITICAL priority "
            f"for immediate relocation consideration. Authorities should review the detailed "
            f"habitation assessments in this report and initiate action plans."
        )
        story.append(Paragraph(msg, styles["Body"]))
    elif n_high > 0:
        msg = (
            f"<b>NOTICE:</b> {n_high} habitation(s) require priority intervention and "
            f"evacuation planning. No immediate relocation is flagged as critical, but "
            f"preparedness actions should be initiated before the next monsoon season."
        )
        story.append(Paragraph(msg, styles["Body"]))
    else:
        story.append(Paragraph(
            "No habitations have been flagged as CRITICAL or HIGH priority at this time. "
            "Routine monitoring is recommended.",
            styles["Body"],
        ))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "<i>Data provenance: Habitation data from OpenStreetMap (or fallback synthetic). "
        "Population figures from OSM population tags only — UNKNOWN where not available. "
        "Safe-area estimates based on hazard grid Low-risk cells. "
        "Road and healthcare distances are straight-line (Euclidean) approximations. "
        "All scores use declared, transparent weights — see PRAVAAH-AI methodology section.</i>",
        styles["Caption"],
    ))
    return story


def _sih_habitation_table(sih_result, styles) -> list:
    """Build the vulnerable habitations ranked table section."""
    story = []
    story.append(Paragraph("PRAVAAH-AI — Vulnerable Habitation Assessment", styles["SectionHead"]))
    story.append(Paragraph(
        "All habitations ranked by relocation priority. "
        "Scores are composite indicators — see methodology for weight declarations.",
        styles["Body"],
    ))
    story.append(Spacer(1, 0.2*cm))

    # Build lookup maps
    exp_map = {e.hab_id: e for e in sih_result.exposure_results}
    vuln_map = {v.hab_id: v for v in sih_result.vulnerability_results}
    cap_map = {c.hab_id: c for c in sih_result.capacity_results}
    rel_list = sih_result.relocation_results  # already sorted by priority

    if not rel_list:
        story.append(Paragraph("No habitation assessments available.", styles["Body"]))
        return story

    PRIORITY_COLORS_RL = {
        "CRITICAL": colors.HexColor("#fdecea"),
        "HIGH":     colors.HexColor("#fef5e7"),
        "MEDIUM":   colors.HexColor("#fef9e7"),
        "LOW":      colors.HexColor("#eafaf1"),
    }
    PRIORITY_TEXT_RL = {
        "CRITICAL": colors.HexColor("#c0392b"),
        "HIGH":     colors.HexColor("#e67e22"),
        "MEDIUM":   colors.HexColor("#d4ac0d"),
        "LOW":      colors.HexColor("#1e8449"),
    }

    hdr = ["#", "Name", "Type", "Hazard", "Vuln.", "Cap.", "Priority", "Population"]
    rows = [hdr]
    for rank, rel in enumerate(rel_list, 1):
        exp = exp_map.get(rel.hab_id)
        vuln = vuln_map.get(rel.hab_id)
        cap = cap_map.get(rel.hab_id)
        pop = (
            f"{exp.population_exposed:,}" if exp and exp.population_source == "osm_tag" and exp.population_exposed
            else "UNKNOWN"
        )
        rows.append([
            str(rank),
            (rel.name or "Unnamed")[:22],
            exp.hab_type if exp else "—",
            f"{exp.hazard_score:.0f}/100" if exp else "—",
            f"{vuln.vulnerability_score:.2f}" if vuln else "—",
            cap.capacity_status[:8] if cap else "—",
            rel.priority_class,
            pop,
        ])

    col_widths = [1*cm, 4.5*cm, 2.5*cm, 2*cm, 1.8*cm, 2*cm, 2.2*cm, 2.5*cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dee2e6")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
    ]
    for i, row in enumerate(rows[1:], 1):
        pc = row[6]
        bg = PRIORITY_COLORS_RL.get(pc, colors.white)
        txt = PRIORITY_TEXT_RL.get(pc, colors.black)
        tbl_style.append(("BACKGROUND", (6, i), (6, i), bg))
        tbl_style.append(("TEXTCOLOR", (6, i), (6, i), txt))
        tbl_style.append(("FONTNAME", (6, i), (6, i), "Helvetica-Bold"))
        if pc == "CRITICAL":
            tbl_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fff5f5")))
        elif i % 2 == 0:
            tbl_style.append(("BACKGROUND", (0, i), (5, i), colors.HexColor("#f8f9fa")))
    tbl.setStyle(TableStyle(tbl_style))
    story.append(tbl)
    return story


def _sih_relocation_section(sih_result, styles) -> list:
    """Build per-habitation detailed relocation priority section."""
    story = []
    story.append(Paragraph("PRAVAAH-AI — Relocation Priority Detail", styles["SectionHead"]))

    # Show detailed breakdown for CRITICAL and HIGH habitations only (PDF space)
    priority_habs = [r for r in sih_result.relocation_results
                     if r.priority_class in ("CRITICAL", "HIGH")]

    if not priority_habs:
        story.append(Paragraph(
            "No CRITICAL or HIGH priority habitations identified in this analysis.",
            styles["Body"],
        ))
        return story

    exp_map = {e.hab_id: e for e in sih_result.exposure_results}
    vuln_map = {v.hab_id: v for v in sih_result.vulnerability_results}
    cap_map = {c.hab_id: c for c in sih_result.capacity_results}

    PRIORITY_BORDER = {
        "CRITICAL": colors.HexColor("#c0392b"),
        "HIGH":     colors.HexColor("#e67e22"),
    }
    PRIORITY_BG = {
        "CRITICAL": colors.HexColor("#fdecea"),
        "HIGH":     colors.HexColor("#fef5e7"),
    }

    for rel in priority_habs[:12]:   # cap at 12 to keep PDF manageable
        exp = exp_map.get(rel.hab_id)
        vuln = vuln_map.get(rel.hab_id)
        cap = cap_map.get(rel.hab_id)

        border_c = PRIORITY_BORDER.get(rel.priority_class, colors.grey)
        bg_c = PRIORITY_BG.get(rel.priority_class, colors.white)

        pop_str = (
            f"{exp.population_exposed:,} (OSM)" if exp and exp.population_source == "osm_tag" and exp.population_exposed
            else "UNKNOWN (not in OSM data)"
        )
        hc_str = f"{cap.nearest_healthcare_km:.1f} km" if cap and cap.nearest_healthcare_km >= 0 else "Not found in area"
        road_str = f"{cap.nearest_road_km:.1f} km" if cap and cap.nearest_road_km >= 0 else "Not found in area"
        safe_str = f"{cap.safe_area_km2:.2f} km²" if cap else "—"
        cap_status = cap.capacity_status if cap else "—"
        vuln_class = vuln.vulnerability_class if vuln else "—"

        factors_text = "\n".join(f"  • {f}" for f in rel.contributing_factors) or "  • No factors identified"

        detail_rows = [
            [Paragraph(f"<b>🏘️ {rel.name or 'Unnamed Habitation'}</b>  —  Priority: <b>{rel.priority_class}</b>  |  Score: {rel.relocation_score:.3f}", styles["Body"]),
             Paragraph(rel.recommended_action, styles["BulletBody"])],
        ]
        detail_table = Table(detail_rows, colWidths=[8*cm, 9*cm])
        detail_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), bg_c),
            ("BOX", (0, 0), (-1, -1), 1.2, border_c),
            ("LINEAFTER", (0, 0), (0, 0), 1, border_c),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        metrics_data = [
            ["Component", "Value", "Class/Status"],
            ["Hazard Score", f"{exp.hazard_score:.1f}/100" if exp else "—",
             exp.hazard_class if exp else "—"],
            ["Vulnerability", f"{vuln.vulnerability_score:.3f}" if vuln else "—",
             vuln_class],
            ["Capacity", f"{cap.capacity_score:.3f}" if cap else "—",
             cap_status],
            ["Population Exposed", pop_str, exp.population_source if exp else "—"],
            ["Safe Area Nearby", safe_str, "—"],
            ["Nearest Healthcare", hc_str, "—"],
            ["Nearest Road", road_str, "—"],
        ]
        metrics_table = Table(metrics_data, colWidths=[4.5*cm, 4.5*cm, 4*cm])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dee2e6")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ]))

        factors_para = Paragraph(
            "<b>Key contributing factors:</b><br/>" +
            "<br/>".join(f"• {f}" for f in rel.contributing_factors),
            styles["BulletBody"],
        )

        story.append(KeepTogether([
            Spacer(1, 0.2*cm),
            detail_table,
            Spacer(1, 0.15*cm),
            metrics_table,
            Spacer(1, 0.1*cm),
            factors_para,
            Spacer(1, 0.3*cm),
        ]))

    if len(sih_result.relocation_results) > 12:
        story.append(Paragraph(
            f"<i>Note: Detailed breakdown shown for top 12 priority habitations. "
            f"Full data for all {len(sih_result.relocation_results)} habitations is available "
            f"in the dashboard and CSV export.</i>",
            styles["Caption"],
        ))

    return story


def export_pdf_report(result, output_path: str | Path, area_name: str = "Study Area", data_tier: int = 3,
                      sih_result=None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _make_styles()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []
    bbox = result.bounding_box
    dist = result.risk_distribution
    grid = result.scored_grid
    total_cells = sum(v for k, v in dist.items() if k != "Water")
    data_tier = result.data_tier
    tier_label = {1: "Tier 1 — Real elevation + rainfall + OSM", 2: "Tier 2 — Partial Real Data", 3: "Tier 3 — Synthetic Data"}
    story.append(Paragraph("PRAVAAH-AI — RISK ASSESSMENT REPORT", styles["Title2"]))
    story.append(Paragraph("Predictive Risk &amp; Vulnerability Assessment for At-Risk Habitations (PRAVAAH-AI)", styles["Subtitle"]))
    story.append(Paragraph(f"{area_name}", styles["Subtitle"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}  |  Data: {tier_label.get(data_tier, 'Unknown')}  |  Method: Weighted susceptibility index", styles["Caption"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a5276"), spaceAfter=12))
    story.append(Paragraph("1. Study Area & Data Summary", styles["SectionHead"]))
    meta_data = [
        ["Parameter", "Value"],
        ["Area Name", area_name],
        ["Bounding Box", f"({bbox.min_lat:.4f}°N, {bbox.min_lon:.4f}°E) to ({bbox.max_lat:.4f}°N, {bbox.max_lon:.4f}°E)"],
        ["Area Coverage", f"{bbox.area_km2:.2f} km²"],
        ["Grid Resolution", f"{result.config.cell_size_meters:.0f}m × {result.config.cell_size_meters:.0f}m cells"],
        ["Total Grid Cells", str(result.cell_count)],
        ["Analysis Duration", f"{result.pipeline_duration_seconds:.1f} seconds"],
        ["Data Source", tier_label.get(data_tier, "Unknown")],
        ["Analysis Method", "Transparent weighted susceptibility index"],
        ["Validation Status", result.analysis_result.validation_note],
        ["Analysis Date", datetime.now().strftime("%d %B %Y")],
    ]
    t = Table(meta_data, colWidths=[6*cm, 11*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
        ("PADDING", (0,0), (-1,-1), 5),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ]))
    story.append(t); story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("2. Grid Cell Data Table (Sample — Top 20 by Risk Score)", styles["SectionHead"]))
    cols_show = ["cell_id", "centroid_lat", "centroid_lon", "elevation_m", "slope_deg", "twi",
                 "rainfall_mean_mm", "dist_water_m", "drainage_capacity", "population_density", "risk_score", "risk_class"]
    available = [c for c in cols_show if c in grid.columns]
    display_names = {"cell_id": "Cell ID", "centroid_lat": "Lat", "centroid_lon": "Lon",
                     "elevation_m": "Elev (m)", "slope_deg": "Slope°", "twi": "TWI",
                     "rainfall_mean_mm": "Rain (mm)", "dist_water_m": "Dist Water (m)",
                     "drainage_capacity": "Drainage", "population_density": "Pop Density",
                     "risk_score": "Risk Score", "risk_class": "Risk Class"}
    sample = grid[grid.risk_class != "Water"].sort_values("risk_score", ascending=False).head(20)
    table_data = [[display_names.get(c, c) for c in available]]
    for _, row in sample.iterrows():
        r = []
        for c in available:
            v = row[c]
            if isinstance(v, float):
                r.append(f"{v:.2f}")
            else:
                r.append(str(v))
        table_data.append(r)
    col_w = [17*cm / len(available)] * len(available)
    dt = Table(table_data, colWidths=col_w, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#dee2e6")),
        ("PADDING", (0,0), (-1,-1), 3),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]
    for i, row in enumerate(table_data[1:], 1):
        rc = row[-1] if available[-1] == "risk_class" else ""
        if rc == "High":
            style_cmds.append(("BACKGROUND", (-1,i), (-1,i), colors.HexColor("#fdecea")))
            style_cmds.append(("TEXTCOLOR", (-1,i), (-1,i), colors.HexColor("#c0392b")))
        elif rc == "Medium":
            style_cmds.append(("BACKGROUND", (-1,i), (-1,i), colors.HexColor("#fef9e7")))
            style_cmds.append(("TEXTCOLOR", (-1,i), (-1,i), colors.HexColor("#d35400")))
        elif rc == "Low":
            style_cmds.append(("BACKGROUND", (-1,i), (-1,i), colors.HexColor("#eafaf1")))
            style_cmds.append(("TEXTCOLOR", (-1,i), (-1,i), colors.HexColor("#1e8449")))
    dt.setStyle(TableStyle(style_cmds))
    story.append(dt); story.append(Spacer(1, 0.3*cm))
    story.append(PageBreak())
    story.append(Paragraph("3. Flood Susceptibility Summary", styles["SectionHead"]))
    summary_data = [["Risk Class", "Cell Count", "Coverage (%)", "Interpretation"]]
    interp = {"High": "Immediate flood risk — low elevation, poor drainage, near water",
              "Medium": "Moderate risk — some vulnerability factors present",
              "Low": "Low susceptibility — good drainage, higher elevation",
              "Water": "Permanent water body — lake, tank, or canal"}
    for rc in ["High", "Medium", "Low", "Water"]:
        if rc in dist:
            cnt = dist[rc]
            pct = f"{cnt/result.cell_count*100:.1f}%" if result.cell_count > 0 else "—"
            summary_data.append([rc, str(cnt), pct, interp.get(rc, "")])
    # Coastal tsunami flag summary row
    n_coastal = int(result.scored_grid.get("is_coastal_tsunami_risk", False).sum()) \
        if "is_coastal_tsunami_risk" in result.scored_grid.columns else 0
    if n_coastal > 0:
        summary_data.append([
            "⚠️ Coastal",
            str(n_coastal),
            f"{n_coastal/result.cell_count*100:.1f}%",
            "Adjacent to ocean/sea — additional tsunami inundation risk",
        ])
    st2 = Table(summary_data, colWidths=[3*cm, 3*cm, 3*cm, 8*cm])
    st2_style = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
        ("PADDING", (0,0), (-1,-1), 6),
        ("ALIGN", (0,0), (2,-1), "CENTER"),
    ]
    for i, row in enumerate(summary_data[1:], 1):
        rc = row[0]
        if rc in RISK_BG:
            st2_style.append(("BACKGROUND", (0,i), (-1,i), RISK_BG[rc]))
            st2_style.append(("TEXTCOLOR", (0,i), (0,i), RISK_COLORS[rc]))
            st2_style.append(("FONTNAME", (0,i), (0,i), "Helvetica-Bold"))
        elif rc == "⚠️ Coastal":
            st2_style.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#fef0e6")))
            st2_style.append(("TEXTCOLOR", (0,i), (0,i), colors.HexColor("#d35400")))
            st2_style.append(("FONTNAME", (0,i), (0,i), "Helvetica-Bold"))
    st2.setStyle(TableStyle(st2_style))
    story.append(st2); story.append(Spacer(1, 0.4*cm))
    chart_buf = _make_chart_buf = _risk_distribution_chart(dist)
    story.append(Image(chart_buf, width=16*cm, height=6*cm))
    story.append(Paragraph("Figure 1: Risk class distribution across all grid cells.", styles["Caption"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("4. Vulnerability Analysis — Why This Area Is at Risk", styles["SectionHead"]))
    reasons = _vulnerability_analysis(result, area_name)
    if reasons:
        for r in reasons:
            story.append(Paragraph(f"• {r}", styles["BulletBody"]))
    else:
        story.append(Paragraph("No significant vulnerability factors identified.", styles["Body"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("5. Factor Weights — Key Flood Susceptibility Drivers", styles["SectionHead"]))
    fi_buf = _feature_importance_chart(result.analysis_result.feature_importances)
    story.append(Image(fi_buf, width=15*cm, height=5.5*cm))
    story.append(Paragraph("Figure 2: Declared index weights — higher values contribute more strongly to the relative score.", styles["Caption"]))
    story.append(Spacer(1, 0.3*cm))
    fi_table_data = [["Feature", "Importance", "Interpretation"]]
    fi_interp = {
        "elevation_m": "Lower elevation → higher flood risk (water flows downhill)",
        "twi": "Higher TWI → terrain collects more water",
        "drainage_capacity": "Lower capacity → water cannot escape quickly",
        "dist_water_m": "Closer to water → first to flood during overflow",
        "rainfall_mean_mm": "Higher rainfall → more water input to the system",
        "slope_deg": "Flatter slope → water pools instead of draining",
        "rainfall_max_24h_mm": "Higher peak rainfall → flash flood potential",
        "population_density": "Higher density → more impervious surfaces, less absorption",
        "aspect_deg": "Terrain orientation affects solar drying and runoff direction",
        "curvature": "Concave terrain (negative curvature) accumulates water",
    }
    for feat, imp in sorted(result.analysis_result.feature_importances.items(), key=lambda x: x[1], reverse=True):
        fi_table_data.append([feat.replace("_", " ").title(), f"{imp:.4f}", fi_interp.get(feat, "—")])
    fi_t = Table(fi_table_data, colWidths=[4.5*cm, 2.5*cm, 10*cm])
    fi_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#dee2e6")),
        ("PADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("ALIGN", (1,0), (1,-1), "CENTER"),
    ]))
    story.append(fi_t)
    story.append(PageBreak())

    # ── Model Performance Metrics ─────────────────────────────────────────────
    ar = result.analysis_result
    story.append(Paragraph("6. Model Performance Metrics", styles["SectionHead"]))

    method_labels = {
        "ensemble": "Ensemble (Weighted Index + Random Forest)",
        "random_forest": "Random Forest (ML)",
        "weighted_susceptibility_index": "Weighted Susceptibility Index (MCDA)",
    }
    story.append(Paragraph(
        f"<b>Model:</b> {method_labels.get(ar.method, ar.method)}",
        styles["Body"],
    ))
    story.append(Paragraph(
        "<b>Training approach:</b> No real flood-event labels are available. "
        "The top 33% of cells by Weighted Susceptibility Index score are used as "
        "high-risk pseudo-labels. Metrics are computed via 5-fold stratified "
        "cross-validation on the same dataset used for scoring.",
        styles["Body"],
    ))
    story.append(Spacer(1, 0.2*cm))

    if ar.method in ("ensemble", "random_forest") and ar.mean_cv_auc is not None:
        # Summary metrics table
        metrics_summary = [
            ["Metric", "Mean (5-fold CV)", "Interpretation"],
            ["AUC-ROC",
             f"{ar.mean_cv_auc:.3f}",
             "Area under ROC curve — 1.0 = perfect, 0.5 = random. Measures ability to rank high-risk cells above low-risk cells."],
            ["F1 Score",
             f"{ar.mean_cv_f1:.3f}" if ar.mean_cv_f1 is not None else "—",
             "Harmonic mean of Precision and Recall — balances false positives and false negatives."],
            ["Accuracy",
             f"{ar.mean_cv_accuracy:.3f}" if ar.mean_cv_accuracy is not None else "—",
             "Fraction of cells correctly classified as high-risk or low-risk."],
            ["Precision",
             f"{ar.mean_cv_precision:.3f}" if ar.mean_cv_precision is not None else "—",
             "Of all cells predicted as high-risk, what fraction actually are? Higher = fewer false alarms."],
            ["Recall (Sensitivity)",
             f"{ar.mean_cv_recall:.3f}" if ar.mean_cv_recall is not None else "—",
             "Of all actual high-risk cells, what fraction were detected? Higher = fewer missed flood zones."],
        ]
        ms_t = Table(metrics_summary, colWidths=[4*cm, 3.5*cm, 9.5*cm])
        ms_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(ms_t)
        story.append(Spacer(1, 0.3*cm))

        # Per-fold CV table
        if ar.cv_auc_scores:
            n_folds = len(ar.cv_auc_scores)
            story.append(Paragraph("Per-Fold Cross-Validation Results", styles["SectionHead"]))
            fold_header = ["Fold", "AUC-ROC", "F1 Score", "Accuracy", "Precision", "Recall"]
            fold_rows = [fold_header]
            for i in range(n_folds):
                fold_rows.append([
                    f"Fold {i+1}",
                    f"{ar.cv_auc_scores[i]:.3f}",
                    f"{ar.cv_f1_scores[i]:.3f}" if ar.cv_f1_scores else "—",
                    f"{ar.cv_accuracy_scores[i]:.3f}" if ar.cv_accuracy_scores else "—",
                    f"{ar.cv_precision_scores[i]:.3f}" if ar.cv_precision_scores else "—",
                    f"{ar.cv_recall_scores[i]:.3f}" if ar.cv_recall_scores else "—",
                ])
            # Mean row
            fold_rows.append([
                "Mean",
                f"{ar.mean_cv_auc:.3f}",
                f"{ar.mean_cv_f1:.3f}" if ar.mean_cv_f1 is not None else "—",
                f"{ar.mean_cv_accuracy:.3f}" if ar.mean_cv_accuracy is not None else "—",
                f"{ar.mean_cv_precision:.3f}" if ar.mean_cv_precision is not None else "—",
                f"{ar.mean_cv_recall:.3f}" if ar.mean_cv_recall is not None else "—",
            ])
            fold_t = Table(fold_rows, colWidths=[2.5*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.8*cm])
            fold_style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.HexColor("#f8f9fa"), colors.white]),
                # Mean row highlighted
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#d4efdf")),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]
            fold_t.setStyle(TableStyle(fold_style))
            story.append(fold_t)
            story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph(
            "<i>Note: Metrics are computed on pseudo-labels derived from the Weighted "
            "Susceptibility Index. They measure how consistently the model identifies "
            "cells that the domain-expert index considers high-risk — not validation "
            "against observed flood events. AUC-ROC > 0.85 indicates strong consistency.</i>",
            styles["Caption"],
        ))
    else:
        story.append(Paragraph(
            "The Weighted Susceptibility Index is a deterministic formula — traditional "
            "ML metrics (AUC, F1, Accuracy) do not apply as there is no classifier or "
            "training process. Run with the Ensemble or Random Forest model to see "
            "cross-validation metrics.",
            styles["Body"],
        ))

    story.append(PageBreak())

    # Risk Zone Parameters section
    for item in _risk_zone_parameters_table(result, styles):
        story.append(item)
    story.append(PageBreak())

    story.append(Paragraph("7. Emergency Services Deployment Plan", styles["SectionHead"]))
    story.append(Paragraph(
        "The following deployment plan is based on the flood risk zonation results. "
        "Resources should be pre-positioned before monsoon onset (June 1) and maintained "
        "through the retreat of the southwest monsoon (October 15).",
        styles["Body"]))
    story.append(Spacer(1, 0.3*cm))
    plan = _emergency_plan(dist, grid)
    for zone_name, zone_stats, actions in plan:
        rc = zone_name.split()[1] if len(zone_name.split()) > 1 else "Low"
        bg = RISK_BG.get(rc, colors.HexColor("#f8f9fa"))
        border = RISK_COLORS.get(rc, colors.grey)
        zone_table = Table(
            [[Paragraph(f"<b>{zone_name}</b>  —  {zone_stats}", styles["Body"]),
              Paragraph(actions.replace("\n", "<br/>"), styles["BulletBody"])]],
            colWidths=[5*cm, 12*cm]
        )
        zone_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,0), bg),
            ("BACKGROUND", (1,0), (1,0), colors.white),
            ("BOX", (0,0), (-1,-1), 1.5, border),
            ("LINEAFTER", (0,0), (0,0), 1, border),
            ("PADDING", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(zone_table); story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Resource Allocation Summary", styles["SectionHead"]))
    high_c = dist.get("High", 0); med_c = dist.get("Medium", 0); low_c = dist.get("Low", 0)
    total_c = high_c + med_c + low_c or 1
    res_data = [
        ["Resource Type", "High Risk Zones", "Medium Risk Zones", "Low Risk Zones"],
        ["Rescue Boats", f"{max(1, high_c//5)}", f"{max(1, med_c//10)}", "0 (staging)"],
        ["Medical Teams", f"{max(2, high_c//8)}", f"{max(1, med_c//15)}", f"{max(1, low_c//20)}"],
        ["Food Packets/day", f"{high_c * 50}", f"{med_c * 20}", f"{low_c * 5}"],
        ["Water Cans/day", f"{high_c * 10}", f"{med_c * 4}", f"{low_c * 1}"],
        ["Evacuation Buses", f"{max(2, high_c//8)}", f"{max(1, med_c//15)}", "0"],
        ["Sandbag Units", f"{high_c * 20}", f"{med_c * 10}", "0"],
        ["Volunteers Needed", f"{max(10, high_c * 2)}", f"{max(5, med_c)}", f"{max(3, low_c//2)}"],
    ]
    res_t = Table(res_data, colWidths=[5*cm, 4*cm, 4*cm, 4*cm])
    res_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND", (1,1), (1,-1), colors.HexColor("#fdecea")),
        ("BACKGROUND", (2,1), (2,-1), colors.HexColor("#fef9e7")),
        ("BACKGROUND", (3,1), (3,-1), colors.HexColor("#eafaf1")),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
        ("PADDING", (0,0), (-1,-1), 6),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (0,-1), [colors.HexColor("#f8f9fa"), colors.white]),
    ]))
    story.append(res_t); story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("8. Recommendations & Next Steps", styles["SectionHead"]))

    # ── PRAVAAH-AI habitation intelligence sections (when sih_result is supplied) ─
    if sih_result is not None:
        story.extend(_sih_executive_summary(sih_result, area_name, styles))
        story.append(PageBreak())
        story.extend(_sih_habitation_table(sih_result, styles))
        story.append(PageBreak())
        story.extend(_sih_relocation_section(sih_result, styles))
        story.append(PageBreak())    # Area-specific recommendations based on actual data
    grid_nw = grid[grid.risk_class != "Water"]
    high_drain = grid_nw[grid_nw.risk_class == "High"]["drainage_capacity"].mean() if dist.get("High", 0) > 0 else 0.5
    high_elev = grid_nw[grid_nw.risk_class == "High"]["elevation_m"].mean() if dist.get("High", 0) > 0 else 0
    high_dist = grid_nw[grid_nw.risk_class == "High"]["dist_water_m"].mean() if dist.get("High", 0) > 0 else 5000
    recs = [
        f"<b>Immediate (Pre-Monsoon):</b> Install water level sensors at the {dist.get('High', 0)} High-risk cells. "
        f"{'Low OSM drainage proxy score in High-risk zones (avg ' + str(round(high_drain, 2)) + '/1.0 — heuristic proxy, not measured hydraulic capacity). Investigate local drainage adequacy on-site.' if high_drain < 0.45 else 'Maintain existing drainage infrastructure.'}",
        f"<b>Short-term (1–3 months):</b> {'Enforce no-construction zones within 200m of water bodies — ' + str(dist.get('Water', 0)) + ' water body cells identified.' if dist.get('Water', 0) > 0 else 'Map local drainage network for improved accuracy.'} "
        f"Upgrade stormwater drainage in High-risk zones.",
        f"<b>Medium-term (3–12 months):</b> Develop community flood early-warning system for {area_name}. "
        f"{'Focus on low-lying areas at ' + str(round(high_elev, 0)) + 'm elevation.' if high_elev > 0 else 'Map underground drainage network.'}",
        "<b>Long-term (1–5 years):</b> Integrate real-time rainfall radar data (IMD Doppler) for dynamic risk updates. "
        "Add soil permeability and land-use layers for improved accuracy.",
        "<b>Data Improvement:</b> Replace synthetic drainage scores with actual municipal stormwater drain network data. "
        "Collect ground-truth flood event data to validate and retrain the ML model.",
    ]
    for rec in recs:
        story.append(Paragraph(f"• {rec}", styles["BulletBody"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6"), spaceAfter=8))
    story.append(Paragraph(
        f"<i>This report was generated by PRAVAAH-AI (Predictive Risk &amp; Vulnerability Assessment "
        f"for At-Risk Habitations) using ML-based geospatial analysis. "
        f"Data sources: NASA SRTM (elevation), OpenStreetMap (water bodies, habitations, roads, healthcare). "
        f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"For emergency use, validate with local BBMP/NDRF authorities.</i>",
        styles["Caption"]))
    doc.build(story)
    return output_path
