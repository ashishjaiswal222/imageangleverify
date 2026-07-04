# 5. Core Logic & Heuristics

The API implements several specific algorithms to determine photo validity. All parameters are configurable in `app/verification/constants.py`.

## 1. Blur Detection
Uses **Laplacian Variance**. The Laplacian operator measures the 2nd derivative of the image, essentially highlighting regions of rapid intensity change (edges). If an image is blurry, it has fewer sharp edges, resulting in a low variance.
- **Logic**: Convert image to grayscale -> Apply Laplacian filter -> Calculate variance.
- **Threshold**: Fails if variance < `45.0`.

## 2. Lighting Detection
Measures the overall brightness of the image to ensure it is not too dark or washed out by a flash.
- **Logic**: Convert image to HSV color space -> Extract the V (Value/Brightness) channel -> Calculate the mean.
- **Threshold**: Fails if mean < `40.0` (Too Dark) or > `230.0` (Too Bright).

## 3. Background Clutter Detection (AI Reference Quality)
Ensures the background is plain and uncluttered, which is essential for AI Image Generators to focus on the subject.
- **Logic**: Uses the `MediaPipe ImageSegmenter` to isolate the person and create a background mask. The Canny edge detection algorithm is applied strictly to the background region to measure "edge density" (busyness).
- **Threshold**: Fails if background edge density > `3.0%`.

## 4. Head Angle Verification (Yaw & Pitch)
Uses 3D Transformation matrices to determine exactly where the user is looking and the camera's relative height.
- **Logic**: MediaPipe extracts a 3x3 rotation matrix for the head. This is converted into Euler angles (Pitch, Yaw, Roll). 
- **Mapping (Yaw)**: 
  - `Front`: absolute Yaw <= 15 degrees.
  - `Right`: Yaw > 25 degrees.
  - `Left`: Yaw < -25 degrees.
- **Eye-Level Constraint (Pitch)**: Fails if absolute Pitch > `20.0` degrees (to prevent high/low-angle perspective distortion).
- **Fallback**: If the face is too small for the Face Mesh to detect landmarks (e.g., a full-body side profile or front profile), the system uses the full-body **Pose Landmarker** as a fallback. It mathematically calculates the ratio of shoulder width vs torso height, and the visibility difference between the left and right ear to prove the rotation. 
  - *Crucial Security Rule*: Even if the AI uses the full-body fallback to calculate the angle, the initial **Face Detector** MUST still detect at least 1 face in the image to pass. If the person's face is completely hidden or off-camera, the image will be rejected with `NO_FACE_DETECTED`.

## 5. Eyes Closed / Blinking
Uses the **Eye Aspect Ratio (EAR)**.
- **Logic**: Calculates the vertical distance between the upper and lower eyelids, divided by the horizontal distance between the eye corners. 
- **Threshold**: If EAR < `0.20`, the eyes are considered closed.

## 6. Neutral Expression Check
Ensures the face is relaxed and not distorted by exaggerated expressions, vital for identity-preserving AI generators.
- **Logic**: Uses `MediaPipe FaceLandmarker` blendshapes (`jawOpen`, `mouthSmileLeft`, `mouthSmileRight`, `eyeBlinkLeft/Right`) to measure facial muscle activation.
- **Threshold**: Fails if `jawOpen` > `0.3`, `mouthSmile` > `0.5`, or asymmetrical blinking > `0.4`.

## 7. Sunglasses & Heavy Occlusion Detection
- **Logic (Primary)**: Checks the eye region for dark contrast (`< 40` intensity) compared to the skin.
- **Logic (Fallback)**: If the face is so heavily obscured by phones, hands, or sunglasses that landmarks completely fail, the system falls back to the `ImageSegmenter`. It analyzes the `InsightFace` bounding box to calculate the ratio of Class 5 (Accessories), Class 4 (Clothes), and Class 2 (Body/Hands) covering the face.
- **Threshold**: Fails if Accessories > `10%` (Eyewear detected) or total occlusion > `20%` (Face Partially Covered).

## 8. Back View Fallback
Since a person facing completely backward has no visible face landmarks, the API relies on body segmentation.
- **Logic**: Uses the Pose Landmarker to verify shoulders are visible. If pose fails, it uses the Image Segmenter to verify that at least 5% of the image contains "Hair".

## 9. Cross-Photo Identity Verification
Uses **Cosine Distance**.
- **Logic**: InsightFace extracts a `1x128` vector representing the unique biometric structure of the face. For the batch endpoint, the API calculates the cosine distance between the `front` vector and the `left`/`right`/`full_body` vectors.
- **Threshold**: If the mathematical distance is > `0.60`, it is a different person.

## 10. Explicit Content Filter (NSFW)
- **Logic**: NudeNet scans the image. If any of the strictly blocked classes (`FEMALE_GENITALIA_EXPOSED`, `BUTTOCKS_EXPOSED`, etc.) are detected with high confidence, the image is immediately rejected.
