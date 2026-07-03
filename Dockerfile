FROM python:3.12-slim as builder

WORKDIR /app

# Install uv
RUN pip install uv

# Install dependencies using uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Required system libraries for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /app/.venv /app/.venv

# Copy application and models
COPY app/ ./app/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY .env.example .env

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
