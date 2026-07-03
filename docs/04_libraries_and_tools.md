# 4. Libraries & Tools

The API is constructed entirely from open-source libraries, ensuring zero vendor lock-in and 100% offline functionality.

### Core Framework
- **FastAPI** (`fastapi`, `uvicorn`): The core web framework. Chosen for its extreme speed and native support for asynchronous Python (`async`/`await`).
- **Pydantic** (`pydantic-settings`): Used for strict type validation of API responses and environment variable management.

### Computer Vision (The Heuristics Engine)
- **OpenCV** (`opencv-python-headless`): The backbone of image processing. It is used to load images into NumPy arrays natively, calculate Laplacian variance (blur detection), ELA variance (heavy editing detection), and convert color spaces for brightness calculation.
- **MediaPipe** (`mediapipe`): Google's lightweight machine learning pipeline. We use specific pre-trained models for this:
  - `FaceDetector` (**`models/blaze_face_short_range.tflite`**): Ensures exactly one face is in the image and checks face centering.
  - `FaceMesh` (**`models/face_landmarker.task`**): Extracting 478 3D facial landmarks to calculate head rotation (Pitch, Yaw, Roll), eye aspect ratio (blinking/closed eyes), and detecting eyewear via iris occlusion.
  - `PoseLandmarker` (**`models/pose_landmarker_heavy.task`**): Extracting 33 full-body landmarks to verify full-body shots and fallback side-profile views.
  - `ImageSegmenter` (**`models/selfie_multiclass_256x256.tflite`**): Extracting background separation masks to ensure the face/hair is not covered by objects or hats.

### Deep Learning (Biometrics & Explicit Filters)
- **InsightFace** (`insightface`): A state-of-the-art 2D and 3D face analysis library. 
  - **Models Used**: Downloads the `buffalo_l` pack (which includes `det_10g.onnx` and `w600k_r50.onnx`) into `~/.insightface`. 
  - **Purpose**: It extracts a 128-dimensional embedding from faces to calculate cosine distance between photos to verify identity. It is also used to enforce photo realism, natively blocking cartoons and robustly counting group photos.
- **NudeNet** (`nudenet`): An ONNX-based classifier.
  - **Models Used**: Downloads `default_onnx_320.onnx` into `~/.NudeNet`.
  - **Purpose**: Scans the image for explicit or NSFW content (e.g., exposed genitalia, buttocks) to prevent inappropriate uploads.
- **ONNXRuntime** (`onnxruntime`): The C++ backend engine used to run the InsightFace and NudeNet models locally on the CPU at high speeds.

### Infrastructure & State
- **Redis** (`redis`): An in-memory data store. Used to temporarily cache facial biometrics during the `/check-single` onboarding flow so they can be cross-checked later.
- **Structlog** (`structlog`): Provides structured JSON logging, crucial for monitoring worker pool status and API timeouts in production.
