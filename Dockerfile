# Optional: lightweight training image
FROM python:3.11-slim

WORKDIR /app

# System deps (git for datasets if needed, plus basic build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt report.md /app/
COPY toxicity_transformer /app/toxicity_transformer
COPY scripts /app/scripts

RUN pip install --no-cache-dir -r requirements.txt

# Default command (override as needed)
CMD ["python", "scripts/train.py", "--max-train-examples", "20000", "--max-val-examples", "5000", "--epochs", "1"]
