# Photo Verification API

A production-grade FastAPI microservice for verifying KYC onboarding photos. Completely open-source, no paid APIs, and no LLMs.

## Quick Start (Local Development)

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv) package manager
- Python 3.12+

### 2. Installation
```bash
uv sync
```

### 3. Download Models
MediaPipe models must be downloaded before running:
```bash
uv run python scripts/download_models.py
```

### 4. Run Server
```bash
uv run uvicorn app.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`. Swagger UI at `http://127.0.0.1:8000/docs`.

### 5. Docker Deployment
```bash
docker-compose up --build
```

For full system architecture and API contracts, see [DOCUMENTATION.md](./DOCUMENTATION.md).
