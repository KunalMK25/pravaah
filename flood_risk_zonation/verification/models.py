# -*- coding: utf-8 -*-
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

ActiveFloodStatus = Literal[
    "ACTIVE_FLOODING",
    "NO_ACTIVE_FLOODING",
    "INSUFFICIENT_EVIDENCE",
    "CHECK_FAILED",
]

@dataclass
class FloodEvidence:
    source: str
    title: str
    location: str
    timestamp: Optional[datetime]
    evidence_text: str
    indicates_active_flooding: bool
    confidence: float
    age_minutes: Optional[float] = None

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class ActiveFloodVerificationResult:
    status: ActiveFloodStatus
    location_name: str
    location_lat: float
    location_lon: float
    verification_timestamp: datetime
    evidence_list: list
    primary_evidence: Optional[FloodEvidence]
    summary: str
    confidence: float
    fallback_reason: Optional[str] = None
    duration_seconds: Optional[float] = None

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.status not in ("ACTIVE_FLOODING", "NO_ACTIVE_FLOODING", "INSUFFICIENT_EVIDENCE", "CHECK_FAILED"):
            raise ValueError(f"Invalid status: {self.status}")

    def is_active_flood_gate(self) -> bool:
        return self.status == "ACTIVE_FLOODING"

    def should_continue_normal_pipeline(self) -> bool:
        return self.status in ("NO_ACTIVE_FLOODING", "INSUFFICIENT_EVIDENCE", "CHECK_FAILED")
