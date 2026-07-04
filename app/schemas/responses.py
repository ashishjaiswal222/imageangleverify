from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List

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
    position: str = Field(description="The position evaluated (e.g., 'front', 'full_body')")
    passed: bool = Field(description="Whether the image passed all verification checks")
    confidence: Optional[float] = Field(None, description="Overall confidence score (0-100) or raw metric")
    primary_reason: Optional[PrimaryReason] = Field(None, description="The primary reason for failure, if any")
    failed_reasons: List[PrimaryReason] = Field(default_factory=list, description="A complete list of all reasons the image failed")
    checks: Dict[str, CheckResult] = Field(description="Detailed results of all executed checks")
    processed_in_ms: int = Field(0, description="Time taken to process this image")

class IdentityConsistency(BaseModel):
    passed: bool
    face_similarity_pairs: Dict[str, float]
    clothing_consistency_score: float
    failed_reasons: List[PrimaryReason] = Field(default_factory=list, description="List of reasons identity consistency failed")

class BatchResults(BaseModel):
    session_id: Optional[str] = None
    overall_passed: bool
    identity_consistency: Optional[IdentityConsistency] = None
    results: Dict[str, VerificationResult | Dict[str, str]]
    processed_in_ms: int = 0
