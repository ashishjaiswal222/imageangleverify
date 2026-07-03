# 3. Endpoints & Frontend Integration

The API exposes two main endpoints designed to support a seamless frontend UI experience.

## How the Frontend Should Send Data
Because the API processes raw image binaries, the frontend **must** send data using the `multipart/form-data` encoding format. 
Base64 encoding is not supported for efficiency reasons.

## Endpoint 1: Real-Time Single Check
**POST** `/api/v1/photo-verify/check-single`

Use this endpoint the moment a user drops a single photo into an upload box on the UI. It provides instant feedback so the user can retake the photo immediately if it's blurry or incorrect.

**Request Payload (multipart/form-data)**
- `file`: The raw binary image file (`image/jpeg`, `image/png`, `image/webp`).
- `position`: The expected angle (`front`, `left`, `right`, `back`, `full_body`).
- `session_id` (optional): A unique string identifying the user's current session. Required if you want the API to cache biometrics for cross-photo identity checking.

**Success Response (200 OK)**
```json
{
  "position": "front",
  "passed": true,
  "confidence": 74.3,
  "primary_reason": null,
  "checks": { ... }
}
```

**Failure Response (200 OK - Heuristic Failure)**
```json
{
  "position": "front",
  "passed": false,
  "primary_reason": {
    "code": "IMAGE_BLURRY",
    "message": "Photo appears blurry or low resolution."
  },
  "checks": { ... }
}
```

---

## Endpoint 2: Final Batch Submission
**POST** `/api/v1/photo-verify/check-batch`

Use this endpoint when the user hits the final "Submit" button. It processes all required photos in parallel and performs the **Identity Consistency** check.

**Request Payload (multipart/form-data)**
- `front`: Raw binary image file.
- `left`: Raw binary image file.
- `right`: Raw binary image file.
- `back`: Raw binary image file.
- `full_body`: Raw binary image file.
- `session_id`: Unique string.

**Response**
The response contains an `overall_passed` boolean, the results for all 5 individual photos, and the massive `identity_consistency` block which mathematically proves if the person in the photos is the same.
```json
{
  "session_id": "xyz123",
  "overall_passed": true,
  "identity_consistency": {
    "passed": true,
    "face_similarity_pairs": {
      "front_left": 0.36,
      "front_right": 0.33
    }
  },
  "results": {
    "front": { ... },
    "left": { ... }
  }
}
```
