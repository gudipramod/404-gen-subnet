"""Real generation pipeline: VLM image analysis -> code-LLM Three.js synthesis
-> local validator loop -> spec-compliant generate.js bytes.
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path
import requests
import torch
from PIL import Image
from loguru import logger
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

VLM_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
CODE_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"

_MINER_REFERENCE_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_MD_PATH = _MINER_REFERENCE_ROOT / "AGENTS.md"
_VALIDATE_JS_PATH = _MINER_REFERENCE_ROOT / "tools" / "validate.js"

MAX_VALIDATION_RETRIES = 3
IMAGE_DOWNLOAD_TIMEOUT = 20

_state = {"vlm": None, "vlm_processor": None, "code_model": None, "code_tokenizer": None}


def load_models() -> None:
    """Load VLM and code-LLM once. Call during pod warmup."""
    logger.info(f"Loading VLM: {VLM_MODEL_ID}")
    _state["vlm_processor"] = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    _state["vlm"] = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    logger.info(f"Loading code LLM: {CODE_MODEL_ID}")
    _state["code_tokenizer"] = AutoTokenizer.from_pretrained(CODE_MODEL_ID)
    _state["code_model"] = AutoModelForCausalLM.from_pretrained(
        CODE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    logger.info("Models loaded and ready")


def _download_image(url: str) -> Image.Image:
    import io
    resp = requests.get(url, timeout=IMAGE_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _analyze_image(image: Image.Image) -> str:
    """VLM pass: describe the object's structure for the code LLM."""
    prompt_text = (
        "Describe this 3D object's structure for someone reconstructing it "
        "from geometric primitives (boxes, cylinders, spheres, cones, tori, "
        "lathes, extrusions). Be precise about: overall shape decomposition "
        "into parts, relative proportions and positions of each part, "
        "approximate colors/materials (metallic, rough, glossy), and any "
        "symmetry or repeated elements. Do not mention textures beyond "
        "solid colors. Keep it factual and structured, under 300 words."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    processor = _state["vlm_processor"]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = _state["vlm"].generate(**inputs, max_new_tokens=500)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    description = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return description.strip()


def _extract_js_code(text: str) -> str:
    """Pull the JS module out of a code-fenced or raw LLM response."""
    match = re.search(r"```(?:js|javascript)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _generate_code(description: str, seed: int, feedback: str | None = None) -> str:
    """Code-LLM pass: structural description -> generate.js source."""
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tokenizer = _state["code_tokenizer"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = _state["code_model"].generate(
            **inputs, max_new_tokens=2000, temperature=0.3, do_sample=True
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
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
