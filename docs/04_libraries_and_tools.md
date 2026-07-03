# 4. Libraries & Tools

The API is constructed entirely from open-source libraries, ensuring zero vendor lock-in and 100% offline functionality.

### Core Framework
- **FastAPI** (`fastapi`, `uvicorn`): The core web framework. Chosen for its extreme speed and native support for asynchronous Python (`async`/`await`).
- **Pydantic** (`pydantic-settings`): Used for strict type validation of API responses and environment variable management.

### Computer Vision (The Heuristics Engine)
- **OpenCV** (`opencv-python-headless`): The backbone of image processing. It is used to load images into NumPy arrays natively, calculate Laplacian variance (blur detection), ELA variance (heavy editing detection), and convert color spaces for brightness calculation.
- **MediaPipe** (`mediapipe`): Google's lightweight machine learning pipeline. Used extensively for:
  - `FaceDetector`: Ensuring exactly one face is in the image.
  - `FaceMesh`: Extracting 468 3D facial landmarks to calculate head rotation (Pitch, Yaw, Roll), eye aspect ratio (blinking/closed eyes), and detecting eyewear via iris occlusion.
  - `PoseLandmarker`: Extracting 33 full-body landmarks to verify full-body shots and fallback side-profile views.
  - `ImageSegmenter`: Extracting background separation masks to ensure the face/hair is not covered by objects or hats.

### Deep Learning (Biometrics & Explicit Filters)
- **InsightFace** (`insightface`): A state-of-the-art 2D and 3D face analysis library. We use the `buffalo_l` model to extract a 128-dimensional embedding from faces. This allows the API to calculate cosine distance between faces in different photos to verify identity.
- **NudeNet** (`nudenet`): An ONNX-based classifier used to scan the image for explicit or NSFW content (e.g., exposed genitalia, buttocks) to prevent inappropriate uploads.
- **ONNXRuntime** (`onnxruntime`): The C++ backend engine used to run the InsightFace and NudeNet models locally on the CPU at high speeds.

### Infrastructure & State
- **Redis** (`redis`): An in-memory data store. Used to temporarily cache facial biometrics during the `/check-single` onboarding flow so they can be cross-checked later.
- **Structlog** (`structlog`): Provides structured JSON logging, crucial for monitoring worker pool status and API timeouts in production.
