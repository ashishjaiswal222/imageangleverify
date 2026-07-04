# 5. Core Logic & Heuristics

The API implements several specific algorithms to determine photo validity. All parameters are configurable in `app/verification/constants.py`.

## The 13 Individual Photo Quality Checks

Every individual photo uploaded to the API undergoes a rigorous gauntlet of 13 parallel checks to ensure it is perfect for AI generation:

1. **Face & Person Detection**: Fails if no face is found, or if multiple faces/group photos are detected. Uses InsightFace and MediaPipe.
2. **Explicit Content**: Scans for NSFW/nudity. Fails if any blocked classes (e.g. exposed genitalia) are detected with high confidence using NudeNet.
3. **Angle & Pitch Match**: Measures 3D Yaw and Pitch to guarantee the user is looking in the correct direction. Includes a tolerance window (±15° yaw for front, ±20° pitch) to allow natural eye-level capture without failing casual selfies.
4. **Blur (Laplacian Variance)**: Fails if the photo is low-resolution or out of focus.
5. **Lighting**: Checks overall brightness. Fails if the image is too dark (e.g., night indoors) or blown out by flash/backlighting.
6. **Uneven Lighting**: Fails if there are harsh shadows on one side of the face (e.g., standing near a single bright window). 
7. **Face Centering**: Fails if the person is standing way off to the edge of the frame.
8. **Eyewear**: Fails if the eyes are covered by dark sunglasses (clear prescription glasses are permitted as they don't block Iris visibility).
9. **Eyes Open**: Fails if the user is blinking or asleep, using Eye Aspect Ratio (EAR).
10. **Occlusion / Face Covered**: Fails if hands, phones, or clothing are heavily blocking the face or hairline.
11. **Neutral Expression**: Fails if winking, shouting, or exaggerated laughing. This check is relaxed to allow soft, natural smiles.
12. **Background Clutter**: Uses Image Segmentation to fail if the background is a messy room, street, or pattern. (Edge density > 3%).
13. **Heavy Editing**: Fails if a Snapchat filter, skin-smoothing beautifier, or AI-art filter was used, by measuring the structural norm of the biometric face embedding.

## The 2 Batch Identity Checks

When submitting the final batch of 5 photos (`/check-batch`), the system guarantees identity consistency:

14. **Cross-Photo Facial Similarity**: Extracts 128-dimensional biometric vectors from the Front, Left, Right, and Full Body photos. It mathematically compares them against each other (Cosine distance < 0.60) to guarantee it is the exact same human being in all photos.
15. **Clothing Consistency**: Scans the clothing colors/patterns of the Front photo and compares it to the Back photo to ensure the user didn't swap outfits or use a stock photo of a back profile.
