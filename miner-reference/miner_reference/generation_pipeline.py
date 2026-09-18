"""Real generation pipeline: VLM image analysis -> code-LLM Three.js synthesis
-> local validator loop -> spec-compliant generate.js bytes.
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path
import base64
import io
import os

import requests
from PIL import Image
from loguru import logger

# Both stages (image analysis + code synthesis) run on the SAME already-running
# vLLM server -- the shared "Quick" tier. It is natively vision-capable, so no
# separate VLM is loaded here: no local weights, no extra VRAM, no second model.
LLM_ENDPOINT = os.environ.get("SN17_LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
LLM_MODEL_NAME = os.environ.get("SN17_LLM_MODEL", "qwen3.6-35b")
LLM_MAX_IMAGE_PX = 768  # downscale before send; keeps prompt tokens sane
# 2000 truncated complex objects mid-file (finish_reason=length -> PARSE_ERROR).
# Measured: a compliant SUV/clock module needs ~3.4k tokens.
LLM_CODE_MAX_TOKENS = int(os.environ.get("SN17_CODE_MAX_TOKENS", "6000"))

_MINER_REFERENCE_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_MD_PATH = _MINER_REFERENCE_ROOT / "AGENTS.md"
_VALIDATE_JS_PATH = _MINER_REFERENCE_ROOT / "tools" / "validate.js"

MAX_VALIDATION_RETRIES = 3
IMAGE_DOWNLOAD_TIMEOUT = 20

_state = {"vlm": None, "vlm_processor": None, "code_model": None, "code_tokenizer": None}


def load_models() -> None:
    """No models are loaded in-process.

    Both pipeline stages call the shared vLLM server (see LLM_ENDPOINT), which
    is already running and serves the rest of the stack. This call only
    verifies it is reachable so warmup fails loudly rather than at round time.
    """
    base = LLM_ENDPOINT.rsplit("/v1/", 1)[0] + "/v1/models"
    try:
        served = [m["id"] for m in requests.get(base, timeout=10).json()["data"]]
    except Exception as exc:
        raise RuntimeError(f"Shared LLM not reachable at {base}: {exc}") from exc
    if LLM_MODEL_NAME not in served:
        raise RuntimeError(f"Model {LLM_MODEL_NAME!r} not served at {base}; has: {served}")
    logger.info(f"Using shared LLM {LLM_MODEL_NAME} at {LLM_ENDPOINT} for vision + code")


def _chat(messages: list, max_tokens: int, temperature: float) -> str:
    """One call into the shared vLLM server, thinking disabled."""
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(LLM_ENDPOINT, json=payload, timeout=240)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _download_image(url: str) -> Image.Image:
    import io
    resp = requests.get(url, timeout=IMAGE_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _analyze_image(image: Image.Image) -> str:
    """Vision pass on the shared LLM: describe structure for the code stage."""
    prompt_text = (
        "Describe this 3D object's structure for someone reconstructing it "
        "from geometric primitives (boxes, cylinders, spheres, cones, tori, "
        "lathes, extrusions). Be precise about: overall shape decomposition "
        "into parts, relative proportions and positions of each part, "
        "approximate colors/materials (metallic, rough, glossy), and any "
        "symmetry or repeated elements. Do not mention textures beyond "
        "solid colors. Keep it factual and structured, under 300 words."
    )
    shrunk = image.copy()
    shrunk.thumbnail((LLM_MAX_IMAGE_PX, LLM_MAX_IMAGE_PX))
    buf = io.BytesIO()
    shrunk.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    content = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt_text},
    ]
    return _chat([{"role": "user", "content": content}], max_tokens=500, temperature=0.3).strip()


def _extract_js_code(text: str) -> str:
    """Pull the JS module out of a code-fenced or raw LLM response.
    Handles: properly closed fences, fences missing a closing marker
    (response cut off), and raw unfenced code. Logs a warning whenever
    it can't find a clean fence, since falling back to raw text risks
    including conversational preamble that breaks JS parsing.
    """
    match = re.search(r"```(?:js|javascript)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    open_match = re.search(r"```(?:js|javascript)?\s*\n(.*)", text, re.DOTALL)
    if open_match:
        logger.warning("Code fence opened but not closed -- using content after opening fence only")
        return open_match.group(1).strip()

    logger.warning(f"No code fence found in response, attempting keyword-based extraction. Raw response start: {text[:200]!r}")
    code_start = re.search(r"^(export\s+default\s+function|function|const|class)\s", text, re.MULTILINE)
    if code_start:
        return text[code_start.start():].strip()

    logger.warning("No fence or code keyword found -- returning raw text as last resort")
    return text.strip()


def _generate_code(description: str, seed: int, feedback: str | None = None) -> str:
    """Code-LLM pass: structural description -> generate.js source.
    Calls the locally-running vLLM server (NVFP4-quantized Qwen3.6-35B-A3B)
    instead of a local transformers model -- ~60x faster on this hardware,
    per benchmarking done 2026-07-30.
    """
    agents_spec = _AGENTS_MD_PATH.read_text()

    system_prompt = (
        "You are a precise Three.js code generator. You must follow every "
        "rule in the specification below exactly -- violations are rejected "
        "by automated validation with no partial credit.\n\n" + agents_spec
    )

    user_prompt = (
        f"Object structural description:\n{description}\n\n"
        f"Deterministic seed (do not use Math.random; if variation is needed, "
        f"derive it algorithmically from this seed value): {seed}\n\n"
        "Write the complete generate.js module now. Output ONLY the code "
        "in a single ```js code block, nothing else."
    )

    if feedback:
        user_prompt += (
            f"\n\nYour previous attempt failed validation with this error:\n"
            f"{feedback}\n\nFix this specific issue and return the complete "
            f"corrected module."
        )

    response = _chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=LLM_CODE_MAX_TOKENS,
        temperature=0.3,
    )
    return _extract_js_code(response)


def _validate_js(js_source: str) -> tuple[bool, str]:
    """Run the local Node.js validator against generated source."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(js_source)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["node", str(_VALIDATE_JS_PATH), "--json", temp_path],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(_MINER_REFERENCE_ROOT),
        )
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            return False, f"validator produced non-JSON output: {result.stdout[:500]} {result.stderr[:500]}"

        passed = parsed.get("passed", False)
        if passed:
            return True, ""
        return False, json.dumps(parsed.get("failures", parsed), indent=2)[:1000]
    except subprocess.TimeoutExpired:
        return False, "validator timed out"
    finally:
        Path(temp_path).unlink(missing_ok=True)


def generate_scene_for_prompt(image_url: str, seed: int) -> bytes:
    """Full pipeline for one prompt: download -> analyze -> generate ->
    validate -> retry-on-failure.
    """
    image = _download_image(image_url)
    description = _analyze_image(image)
    logger.info(f"VLM description: {description[:150]}...")

    feedback = None
    for attempt in range(1, MAX_VALIDATION_RETRIES + 1):
        js_source = _generate_code(description, seed, feedback=feedback)
        passed, error_detail = _validate_js(js_source)

        if passed:
            logger.info(f"Validation passed on attempt {attempt}")
            return js_source.encode("utf-8")

        logger.warning(f"Validation failed on attempt {attempt}: {error_detail[:200]}")
        feedback = error_detail

    raise RuntimeError(
        f"Failed to produce valid module after {MAX_VALIDATION_RETRIES} attempts. "
        f"Last error: {feedback}"
    )


import os
import boto3
from botocore.config import Config

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "ac27d7f3a7bc16bb086351e7f520d56b")
R2_BUCKET = os.environ.get("R2_BUCKET", "node6-sn17")
R2_PUBLIC_URL_BASE = os.environ.get("R2_PUBLIC_URL_BASE", "https://pub-1d1eece1ca024d94b6b52cf0e7945c72.r2.dev")

def _get_r2_client():
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_to_r2(stem: str, js_bytes: bytes, round_id: str) -> str:
    """Upload a generated .js file to R2, return its public CDN URL."""
    client = _get_r2_client()
    key = f"rounds/{round_id}/{stem}.js"
    client.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=js_bytes,
        ContentType="application/javascript",
    )
    return f"{R2_PUBLIC_URL_BASE}/{key}"
