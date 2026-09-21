FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and data assets
COPY src/ ./src/
COPY data/ ./data/

# Install application package
RUN pip install -e .

# Default entrypoint for interactive chat
ENTRYPOINT ["python", "-m", "studyagent.cli"]
CMD []
