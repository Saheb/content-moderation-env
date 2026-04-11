FROM python:3.11-slim

# Install system dependencies and cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory to the standard Hugging Face directory
WORKDIR /code

# Copy requirements first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies with uv for faster builds
RUN pip install --no-cache-dir uv && \
    uv sync --frozen && \
    uv pip install --no-cache-dir -e .

# Copy the rest of the application
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /code
USER app

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run Uvicorn with optimized settings for production
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout", "300", \
     "--timeout-keep-alive", "65", \
     "--log-level", "info"]
