import time
import mediapipe as mp
from typing import Dict, Any, List

from app.utils.image_io import validate_and_decode_image
from app.schemas.responses import VerificationResult, CheckResult, PrimaryReason
from app.verification.person_checks import check_person_and_face
from app.verification.angle_detector import check_head_pose
from app.verification.quality_checks import check_blur, check_lighting, check_face_centering, check_explicit_content
from app.verification.occlusion_checks import check_eyes_open, check_eyewear, check_face_coverage, check_neutral_expression
from app.verification.background_checks import check_background_clutter
from app.verification.edit_detector import check_heavy_editing
from app.utils.errors import VerificationError, ReasonCode

def verify_image(file_bytes: bytes, content_type: str, position: str) -> VerificationResult:
    start_time = time.time()
    checks: Dict[str, CheckResult] = {}
    primary_reason = None
    failed_reasons: List[PrimaryReason] = []
    overall_passed = True

    def record_check(name: str, passed: bool, score: float = None, reason_code: str = None, message: str = None, details: Dict[str, Any] = None):
        nonlocal overall_passed, primary_reason, failed_reasons
        checks[name] = CheckResult(
            passed=passed,
            score=score,
            message=message,
            details=details
        )
        if not passed:
            overall_passed = False
            if reason_code is not None:
                new_reason = PrimaryReason(code=reason_code, message=message)
                failed_reasons.append(new_reason)
                # Keep the first failure as primary
                if primary_reason is None:
                    primary_reason = new_reason

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
        else:
            is_cartoon = any(r.code == ReasonCode.CARTOON_OR_ART_DETECTED.value for r in failed_reasons)
            if checks["person_and_face"].passed or is_cartoon or position == "back":
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

                # 5.5 Uneven Lighting
                from app.verification.quality_checks import check_uneven_lighting
                passed, score, code, msg = check_uneven_lighting(mp_image, image_np)
                record_check("uneven_lighting", passed, score, code, msg)


                if position in ["front", "left", "right", "full_body"]:
                    # 6. Face Centering
                    passed, score, code, msg = check_face_centering(mp_image, image_np)
                    record_check("face_centered", passed, score, code, msg)

                    # 7. Eyewear
                    passed, score, code, msg = check_eyewear(mp_image, image_np)
                    record_check("eyewear", passed, score, code, msg)

                    # 8. Eyes Open (Bypass for full_body because distant eyes cannot be reliably measured)
                    if position == "full_body":
                        passed, score, code, msg = True, None, None, None
                    else:
                        passed, score, code, msg = check_eyes_open(mp_image, image_np)
                    record_check("eyes_open", passed, score, code, msg)

                    # 9. Face Coverage
                    passed, score, code, msg = check_face_coverage(mp_image, image_np)
                    record_check("face_covered", passed, score, code, msg)
                    
                    # 9.5 Neutral Expression
                    passed, score, code, msg = check_neutral_expression(mp_image, image_np)
                    record_check("neutral_expression", passed, score, code, msg)

                # 9.8 Background Clutter Check
                passed, score, code, msg = check_background_clutter(mp_image, image_np)
                record_check("background_cluttered", passed, score, code, msg)

                # 10. Heavy Editing
                passed, score, code, msg = check_heavy_editing(image_np)
                record_check("heavy_editing", passed, score, code, msg)

        # Priority Override: If a face has sunglasses, it often causes a massive anomaly that triggers the Cartoon error.
        # We must prioritize EYEWEAR_DETECTED and clear the false Cartoon error.
        if "eyewear" in checks and not checks["eyewear"].passed:
            if primary_reason and primary_reason.code == ReasonCode.CARTOON_OR_ART_DETECTED.value:
                primary_reason = PrimaryReason(code=ReasonCode.EYEWEAR_DETECTED.value, message=checks["eyewear"].message)
                checks["person_and_face"].passed = True
                checks["person_and_face"].message = None
                failed_reasons = [r for r in failed_reasons if r.code != ReasonCode.CARTOON_OR_ART_DETECTED.value]

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
        failed_reasons=failed_reasons,
        checks=checks,
        processed_in_ms=processed_in_ms
    )
