"""Round generation runner for 404-GEN SN17.

Runs under .venv-vllm (torch + transformers + PIL + loguru + requests +
boto3). round_driver.py runs under .venv-cli (bittensor 10.4.0, no torch)
and shells out here for the generate+upload step. Round 39 died doing this
in-process under .venv-cli with "No module named 'torch'".

v3 (2026-09-07, r42 + 09-07 crash-loop postmortem) - reliability changes:
  - PER-PROMPT save + upload: each .js is written locally and pushed to R2
    the moment it validates, instead of everything at the very end. If this
    process dies mid-round, R2 already holds every completed prompt, so the
    snapshot taken at the generation->downloading transition is a real
    partial round (scored per-prompt; per-prompt nulls are tracked in
    submitted.json) instead of empty content. Round 42 lost exactly this
    way: all-at-end save+upload landed ~2h past the generation deadline and
    the snapshot came back empty -> the -1.0 auto-loss sentinel.
  - GEN LOCK (logs/.gen_lock_<n>): one live generation per round, with a
    heartbeat refreshed every prompt. Round 42's driver crash storm
    restarted the driver 9 times; every restart spawned a NEW gen while
    orphaned old ones kept running, so two ~16GB VLMs sat on the GPU at
    once and NVRM OOMs compounded. A second attempt on the same round now
    exits 4 instead of starting a second model stack.
  - PREFLIGHT: vLLM /health + host memory (MemAvailable/SwapFree) checked
    before the ~16GB VLM load, with bounded backoff instead of dying
    mid-load when the box is squeezed.

The runner: fetches seed/prompts -> for each prompt: generates .js, saves
to round_<n>_output/, uploads to R2, refreshes the lock heartbeat -> writes
the logs/driver_uploaded_<n> marker when at least one result exists. The
driver verifies the marker afterwards.

Exit codes: 2 = bad/missing R2 creds, 3 = zero successful generations,
4 = another live generation holds this round, 5 = preflight gave up.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOGS = BASE.parent / "logs"
sys.path.insert(0, str(BASE))

import orchestrate_round as orc

VLLM_HEALTH_URL = "http://localhost:8000/health"
PREFLIGHT_MIN_AVAIL_GIB = 20   # ~16GB VLM load + working set; box is 121Gi
# 0: this node's 16 GiB swap is persistently full (SwapFree ~0) at baseline;
# it is not an OOM signal here. MemAvailable is the gate that matters on a
# 121Gi unified-memory GB10. (2026-09-07: a 1 GiB gate would have made every
# preflight fail -> no generation at all.)
PREFLIGHT_MIN_SWAP_GIB = 0
PREFLIGHT_MAX_TRIES = 15
PREFLIGHT_RETRY_SEC = 120
LOCK_MAX_AGE_SEC = 1800        # heartbeat older than this = stale lock


def log(msg):
    print("[gen] " + msg, flush=True)


def lock_path(n):
    return LOGS / (".gen_lock_%d" % n)


def acquire_lock(n):
    """Take the per-round gen lock; returns True if we hold it.

    A live holder (pid alive, heartbeat fresh) means another generation is
    running for this round -> the caller must exit, not join. A dead or
    stale holder is an orphan from a previous crash -> log it and take over.
    """
    lp = lock_path(n)
    if lp.exists():
        holder = -1
        try:
            holder = int(json.loads(lp.read_text()).get("pid", -1))
        except Exception:
            pass
        age = int(time.time() - lp.stat().st_mtime)
        alive = False
        if holder > 0:
            try:
                os.kill(holder, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
        if alive and age < LOCK_MAX_AGE_SEC:
            log("round %d: gen lock held by LIVE pid %d (heartbeat %ds ago < %ds) - refusing to start a second generation"
                % (n, holder, age, LOCK_MAX_AGE_SEC))
            return False
        log("round %d: taking over stale gen lock (holder pid=%s alive=%s, heartbeat %ds old)"
            % (n, holder, alive, age))
    lp.write_text(json.dumps({"pid": os.getpid(),
                              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    return True


def refresh_lock(n):
    """Heartbeat: rewrite the lock with the current time (every prompt)."""
    try:
        data = {"pid": os.getpid()}
        lp = lock_path(n)
        if lp.exists():
            try:
                data.update(json.loads(lp.read_text()))
            except Exception:
                pass
        data["pid"] = os.getpid()
        data["last_heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        lp.write_text(json.dumps(data) + "\n")
    except Exception as e:
        log("lock heartbeat failed: %s" % e)


def _meminfo_gib(prefix):
    try:
        for line in open("/proc/meminfo"):
            if line.startswith(prefix):
                return int(line.split()[1]) // (1024 * 1024)
    except Exception:
        pass
    return 0


def preflight():
    """vLLM up + enough host headroom before loading the VLM. Bounded wait.

    On r42 the VLM load hit NVRM OOM while the GPU pool was saturated by
    co-tenants; waiting for headroom and retrying is strictly better than
    dying mid-load and orphaning the round.
    """
    for attempt in range(1, PREFLIGHT_MAX_TRIES + 1):
        ok_vllm = False
        try:
            with urllib.request.urlopen(VLLM_HEALTH_URL, timeout=10) as r:
                ok_vllm = (r.status == 200)
        except Exception:
            ok_vllm = False
        avail = _meminfo_gib("MemAvailable:")
        swap = _meminfo_gib("SwapFree:")
        if ok_vllm and avail >= PREFLIGHT_MIN_AVAIL_GIB and swap >= PREFLIGHT_MIN_SWAP_GIB:
            log("preflight ok: vllm=200 MemAvailable=%dGiB SwapFree=%dGiB (attempt %d/%d)"
                % (avail, swap, attempt, PREFLIGHT_MAX_TRIES))
            return True
        log("preflight not ready (attempt %d/%d): vllm_ok=%s MemAvailable=%dGiB SwapFree=%dGiB "
            "- retrying in %ds" % (attempt, PREFLIGHT_MAX_TRIES, ok_vllm, avail, swap, PREFLIGHT_RETRY_SEC))
        time.sleep(PREFLIGHT_RETRY_SEC)
    log("preflight gave up after %d tries" % PREFLIGHT_MAX_TRIES)
    return False


def run(n, seed, prompts, limit=None):
    """Core loop: per prompt, generate -> save locally -> upload -> heartbeat.

    limit: optional cap on the number of prompts (test harnesses only).
    Returns (results, failed) - same shape as orchestrate_round.generate_all.
    """
    from miner_reference.generation_pipeline import (
        load_models, generate_scene_for_prompt, upload_to_r2)

    log("loading local VLM (~16GB) + connecting to vLLM code-LLM ...")
    load_models()
    refresh_lock(n)

    local = BASE / ("round_%d_output" % n)
    local.mkdir(exist_ok=True)
    todo = prompts[:limit] if limit else prompts
    results = {}
    failed = {}
    for i, prompt in enumerate(todo, 1):
        stem = prompt["stem"]
        try:
            js_bytes = generate_scene_for_prompt(prompt["image_url"], seed)
        except Exception as e:
            failed[stem] = str(e)
            log("%d/%d %s FAILED: %s" % (i, len(todo), stem, e))
            refresh_lock(n)
            continue
        try:
            with open(str(local / (stem + ".js")), "wb") as f:
                f.write(js_bytes)
        except OSError as e:
            failed[stem] = "local save failed: %s" % e
            log("%d/%d %s save FAILED: %s" % (i, len(todo), stem, e))
            refresh_lock(n)
            continue
        try:
            url = upload_to_r2(stem, js_bytes, round_id=str(n))
        except Exception as e:
            # File is safe locally; keep it out of the upload results so the
            # round isn't marked uploaded with a hole, but keep the gen going.
            failed[stem] = "r2 upload failed: %s" % e
            log("%d/%d %s upload FAILED: %s" % (i, len(todo), stem, e))
            refresh_lock(n)
            continue
        results[stem] = js_bytes
        refresh_lock(n)
        log("%d/%d %s ok -> %s" % (i, len(todo), stem, url))
    return results, failed


def main():
    n = int(sys.argv[1])
    orc._load_r2_credentials_from_file()
    missing = [v for v in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not os.environ.get(v)]
    if missing:
        log("FATAL: missing R2 credentials: %s" % missing)
        sys.exit(2)
    ak = os.environ["R2_ACCESS_KEY_ID"]
    sk = os.environ["R2_SECRET_ACCESS_KEY"]
    if len(ak) != 32 or len(sk) != 64:
        log("FATAL: R2 credentials wrong length (access=%d, secret=%d)" % (len(ak), len(sk)))
        sys.exit(2)
    if not acquire_lock(n):
        # Another live generation owns this round. Do NOT join: a second
        # ~16GB VLM on a squeezed GPU is what made r42's NVRM storm worse.
        sys.exit(4)
    if not preflight():
        sys.exit(5)
    seed, prompts = orc.fetch_round_data(n)
    log("round %d: seed=%s prompts=%d" % (n, seed, len(prompts)))
    results, failed = run(n, seed, prompts)
    if not results:
        log("FATAL: no successful generations; failed=%s" % failed)
        sys.exit(3)
    log("done: %d ok, %d failed" % (len(results), len(failed)))
    (LOGS / ("driver_uploaded_%d" % n)).write_text("done\n")
    log("marker written: driver_uploaded_%d" % n)


if __name__ == "__main__":
    main()
