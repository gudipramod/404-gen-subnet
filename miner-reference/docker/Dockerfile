# Verification image. Built for linux/amd64 and run by the orchestrator on a
# 4xH200 (Hopper) pod -- NOT on our Blackwell/aarch64 nodes. Nothing here may
# depend on NVFP4 (Blackwell-only) or on reaching our cluster: the pod is
# isolated, so the image carries its own model servers.
#
# Build context is the repository root:  docker build -f miner-reference/docker/Dockerfile .
FROM vllm/vllm-openai:v0.11.0

# Node.js 20+ for the local validator (miner-reference/validator)
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps for the miner API (torch/vLLM already in the base image)
COPY miner-reference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Validator (Node) deps -- npm ci, because the committed node_modules is
# incomplete (.gitignore strips dist/) and npm install would report "up to date"
COPY miner-reference/validator/package.json miner-reference/validator/package-lock.json ./validator/
RUN cd validator && npm ci --omit=dev

# Models are NOT baked in. Baking ~30GB onto the vLLM base exhausted the GitHub
# runner's disk (build killed at 44min). The spec counts model downloads inside
# the 4h ready budget -- "image pull is not counted, but model downloads and
# warm-up are" -- so the entrypoint fetches them at pod start, where bandwidth
# is real. Revisions stay pinned for reproducibility.
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/models/hf \
    SN17_VISION_REPO=Qwen/Qwen2-VL-7B-Instruct \
    SN17_VISION_REV=eed13092ef92e448dd6875b2a00151bd3f7db0ac \
    SN17_CODE_REPO=Qwen/Qwen2.5-Coder-14B-Instruct \
    SN17_CODE_REV=aedcc2d42b622764e023cf882b6652e646b95671
RUN pip install --no-cache-dir "huggingface_hub[cli]==1.23.0"

COPY miner-reference/ ./
RUN chmod +x docker/entrypoint.sh

EXPOSE 10006
CMD ["./docker/entrypoint.sh"]
