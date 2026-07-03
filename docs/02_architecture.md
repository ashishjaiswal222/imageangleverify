# 2. API Architecture

The API is built as an asynchronous Python microservice heavily optimized for Computer Vision throughput.

## High-Level Architecture
1. **FastAPI Web Layer**: Handles incoming HTTP requests and multipart form data. It instantly validates file size, format, and structure.
2. **Pre-Processing Engine**: Decodes the uploaded images natively via OpenCV (ignoring standard web wrappers for speed).
3. **Parallel Worker Pool**: Computer Vision tasks are heavily CPU-bound and block the Python GIL (Global Interpreter Lock). To bypass this, the API utilizes a `ProcessPoolExecutor`. Incoming batches of photos (up to 5) are immediately delegated to independent worker processes that execute the AI models simultaneously.
4. **AI/ML Layer (In-Memory)**: A robust `models_loader` pre-loads the ONNX and MediaPipe models into RAM during server boot. This eliminates "cold starts" during inference.
5. **Redis Cache (State Management)**: Stores temporary biometric embeddings (with a strict TTL of 1 hour) allowing the frontend to verify photos one-by-one in real-time, then tying them together for identity validation at the end.

## The Processing Pipeline
Every photo passes through a structured, fail-fast pipeline:
1. **Decoding & Format Check**: (`image_io.py`)
2. **Explicit Content Check**: Fails immediately if NSFW.
3. **Person & Face Check**: Ensures exact face counts (no group photos).
4. **Quality Checks**: Lighting, Blur, and ELA (Heavy Editing).
5. **Occlusion Checks**: Sunglasses and Hair blocking face.
6. **Angle Verification**: Uses 3D rotation matrices to verify head pose.
7. **Biometric Extraction**: Extracts 128-d embeddings if the photo passes all the above.

## Scale & Hardware
Because all processing is done entirely offline (no external API calls), the architecture relies completely on the host machine's CPU/RAM. The `BATCH_TIMEOUT_SECONDS` config handles slow consumer CPUs by allowing up to 45 seconds for a massive parallel batch to complete.
