FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Node.js 20+ required at runtime for the local validator (tools/validate.js)
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps -- pinned exact versions per subnet requirement
COPY miner-reference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Validator (Node) deps
COPY miner-reference/validator/package.json miner-reference/validator/package-lock.json ./validator/
RUN cd validator && npm ci --omit=dev

# Application code
COPY miner-reference/ ./

EXPOSE 10006
CMD ["python", "-m", "uvicorn", "miner_reference.service:app", "--host", "0.0.0.0", "--port", "10006"]
