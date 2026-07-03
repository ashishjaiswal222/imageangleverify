import time
import mediapipe as mp
from typing import Dict, Any

from app.utils.image_io import validate_and_decode_image
from app.schemas.responses import VerificationResult, CheckResult, PrimaryReason
from app.verification.person_checks import check_person_and_face
from app.verification.angle_detector import check_head_pose
from app.verification.quality_checks import check_blur, check_lighting, check_face_centering, check_explicit_content
from app.verification.occlusion_checks import check_eyes_open, check_eyewear, check_face_coverage
from app.verification.edit_detector import check_heavy_editing
from app.utils.errors import VerificationError, ReasonCode

def verify_image(file_bytes: bytes, content_type: str, position: str) -> VerificationResult:
    start_time = time.time()
    checks: Dict[str, CheckResult] = {}
    primary_reason = None
    overall_passed = True

    def record_check(name: str, passed: bool, score: float = None, reason_code: str = None, message: str = None, details: Dict[str, Any] = None):
        nonlocal overall_passed, primary_reason
        checks[name] = CheckResult(
            passed=passed,
            score=score,
            message=message,
            details=details
        )
        if not passed:
            overall_passed = False
            # Keep the first failure as primary
            if primary_reason is None and reason_code is not None:
                primary_reason = PrimaryReason(code=reason_code, message=message)

    try:
        # 1. Decode & Sanity
        image_np = validate_and_decode_image(file_bytes, content_type)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        
        # 2. Person & Face check
        passed, score, code, msg = check_person_and_face(mp_image, position)
        record_check("person_and_face", passed, score, code, msg)
        
        # 2.5 Explicit Content Check
        passed, score, code, msg = check_explicit_content(image_np)
        record_check("explicit_content", passed, score, code, msg)

        # Stop early if explicit
        if not passed:
            overall_passed = False
        elif checks["person_and_face"].passed or position == "back": # If back, face might not be detected but we continue to check pose
            # 3. Angle Check
            if position != "full_body":
                passed, score, code, msg, detected, expected = check_head_pose(mp_image, position)
                record_check("angle_match", passed, score, code, msg, details={"detected_angle": detected, "expected_angle": expected})

            # 4. Blur
            passed, score, code, msg = check_blur(image_np)
            record_check("blur", passed, score, code, msg)

            # 5. Lighting
            passed, score, code, msg = check_lighting(image_np)
            record_check("lighting", passed, score, code, msg)

            if position in ["front", "left", "right"]:
                # 6. Face Centering
                passed, score, code, msg = check_face_centering(mp_image, image_np)
                record_check("face_centered", passed, score, code, msg)

                # 7. Eyewear
                passed, score, code, msg = check_eyewear(mp_image, image_np)
                record_check("eyewear", passed, score, code, msg)

                # 8. Eyes Open
                passed, score, code, msg = check_eyes_open(mp_image)
                record_check("eyes_open", passed, score, code, msg)

                # 9. Face Coverage
                passed, score, code, msg = check_face_coverage(mp_image, image_np)
                record_check("face_covered", passed, score, code, msg)

            # 10. Heavy Editing
            passed, score, code, msg = check_heavy_editing(image_np)
            record_check("heavy_editing", passed, score, code, msg)

    except VerificationError as e:
        if e.code in [ReasonCode.FILE_TOO_LARGE.value, ReasonCode.UNSUPPORTED_FORMAT.value, ReasonCode.CORRUPT_IMAGE.value, ReasonCode.RESOLUTION_TOO_LOW.value]:
            raise
        overall_passed = False
        primary_reason = PrimaryReason(code=e.code.value, message=e.message)
    except Exception as e:
        overall_passed = False
        primary_reason = PrimaryReason(code="INTERNAL_ERROR", message=f"An unexpected error occurred: {str(e)}")

    # Calculate aggregate confidence
    confidence_scores = [c.score for c in checks.values() if c.passed and c.score is not None]
    avg_confidence = None
    if confidence_scores:
        avg_confidence = round(sum(confidence_scores) / len(confidence_scores), 4)

    processed_in_ms = int((time.time() - start_time) * 1000)

    return VerificationResult(
        position=position,
        passed=overall_passed,
        confidence=avg_confidence,
        primary_reason=primary_reason,
        checks=checks,
        processed_in_ms=processed_in_ms
    )
