from pydantic import BaseModel, Field
from typing import Dict, Optional, Any

class CheckResult(BaseModel):
    passed: bool
    score: Optional[float] = None
    message: Optional[str] = None
    # Flexible extra metadata (e.g. detected_angle, expected_angle)
    details: Optional[Dict[str, Any]] = None

class PrimaryReason(BaseModel):
    code: str
    message: str

class VerificationResult(BaseModel):
    position: str
    passed: bool
    confidence: Optional[float] = None
    primary_reason: Optional[PrimaryReason] = None
    checks: Dict[str, CheckResult]
    processed_in_ms: int = 0

class IdentityConsistency(BaseModel):
    passed: bool
    face_similarity_pairs: Dict[str, float]
    clothing_consistency_score: float
    reason_code: Optional[str] = None
    message: Optional[str] = None

class BatchResults(BaseModel):
    session_id: Optional[str] = None
    overall_passed: bool
    identity_consistency: Optional[IdentityConsistency] = None
    results: Dict[str, VerificationResult | Dict[str, str]]
    processed_in_ms: int = 0
