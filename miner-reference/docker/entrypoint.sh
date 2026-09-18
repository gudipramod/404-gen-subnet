#!/usr/bin/env bash
# Verification-pod entrypoint.
#
# The orchestrator runs this image on a 4xH200 pod that has NO route back to
# our cluster, so the image carries its own model servers. Two vLLM servers are
# started on loopback and the FastAPI miner talks to them over the same
# OpenAI-compatible HTTP the local deployment uses -- identical code path.
set -euo pipefail

VISION_MODEL=${SN17_VISION_MODEL_PATH:-/models/qwen2-vl-7b-instruct}
CODE_MODEL=${SN17_CODE_MODEL_PATH:-/models/qwen2.5-coder-14b-instruct}

echo "[entrypoint] starting vision server (:8000)"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$VISION_MODEL" --served-model-name vision \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --gpu-memory-utilization 0.42 \
  --max-model-len 8192 &

echo "[entrypoint] starting code server (:8001)"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$CODE_MODEL" --served-model-name code \
  --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --gpu-memory-utilization 0.42 \
  --max-model-len 16384 &

# The 4h ready budget covers warmup, so wait properly rather than racing.
for port in 8000 8001; do
  echo "[entrypoint] waiting for :$port"
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && break
    sleep 5
  done
  curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 \
    || { echo "[entrypoint] FATAL: :$port never became ready"; exit 1; }
  echo "[entrypoint] :$port ready"
done

export SN17_LLM_ENDPOINT="http://127.0.0.1:8000/v1/chat/completions"
export SN17_LLM_MODEL="vision"
export SN17_CODE_ENDPOINT="http://127.0.0.1:8001/v1/chat/completions"
export SN17_CODE_MODEL="code"

echo "[entrypoint] starting miner API on 0.0.0.0:10006"
exec python3 -m uvicorn miner_reference.service:app --host 0.0.0.0 --port 10006
