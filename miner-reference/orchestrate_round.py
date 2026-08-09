"""Round orchestration for 404-GEN SN17: polls for MINER_GENERATION stage,
fetches prompts, runs generation, uploads to R2, commits on-chain.
"""
import json
import time
import subprocess
import sys
from pathlib import Path

import requests
import bittensor as bt

STATE_URL = "https://raw.githubusercontent.com/404-Repo/404-active-competition/main/state.json"
ROUND_URL_TMPL = "https://raw.githubusercontent.com/404-Repo/404-active-competition/main/rounds/{round}/{file}"

WALLET_NAME = "my_chrome_wallet"
WALLET_HOTKEY = "gen404_hotkey"
NETUID = 17
NETWORK = "finney"

GITHUB_REPO = "gudipramod/404-gen-subnet"
R2_PUBLIC_URL_BASE = "https://pub-1d1eece1ca024d94b6b52cf0e7945c72.r2.dev"

POLL_INTERVAL_SECONDS = 300  # check every 5 min while waiting for stage change

MINER_REF_DIR = Path(__file__).resolve().parent


def get_state() -> dict:
    resp = requests.get(STATE_URL, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_round_file(round_num: int, filename: str) -> requests.Response:
    url = ROUND_URL_TMPL.format(round=round_num, file=filename)
    return requests.get(url, timeout=15)


def get_latest_commit_sha() -> str:
    """Get the current HEAD commit SHA of this local git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=MINER_REF_DIR.parent,  # repo root, not miner-reference/
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def wait_for_miner_generation_stage() -> int:
    """Poll state.json until stage == 'miner_generation'. Returns round number."""
    print("Polling for MINER_GENERATION stage...", flush=True)
    while True:
        state = get_state()
        print(f"  current_round={state['current_round']} stage={state['stage']}", flush=True)
        if state["stage"] == "miner_generation":
            return state["current_round"]
        time.sleep(POLL_INTERVAL_SECONDS)


def fetch_round_data(round_num: int) -> tuple[int, list[dict]]:
    """Fetch seed and prompts for the round. Returns (seed, prompt_list)."""
    seed_resp = get_round_file(round_num, "seed.json")
    seed_resp.raise_for_status()
    seed = seed_resp.json()["seed"]

    prompts_resp = get_round_file(round_num, "prompts.txt")
    prompts_resp.raise_for_status()
    urls = [line.strip() for line in prompts_resp.text.splitlines() if line.strip()]

    prompts = []
    for url in urls:
        stem = Path(url).stem
        prompts.append({"stem": stem, "image_url": url})

    print(f"Round {round_num}: seed={seed}, {len(prompts)} prompts", flush=True)
    return seed, prompts


def generate_all(prompts: list[dict], seed: int, round_num: int) -> tuple[dict, dict]:
    """Run generation for every prompt. Returns (results dict stem->bytes, failed dict stem->reason)."""
    sys.path.insert(0, str(MINER_REF_DIR))
    from miner_reference.generation_pipeline import load_models, generate_scene_for_prompt

    print("Loading models...", flush=True)
    load_models()

    results = {}
    failed = {}
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generating {prompt['stem']}...", flush=True)
        try:
            js_bytes = generate_scene_for_prompt(prompt["image_url"], seed)
            results[prompt["stem"]] = js_bytes
            print(f"  OK ({len(js_bytes)} bytes)", flush=True)
        except Exception as e:
            failed[prompt["stem"]] = str(e)
            print(f"  FAILED: {e}", flush=True)

    print(f"Generation complete: {len(results)} succeeded, {len(failed)} failed", flush=True)
    return results, failed


def upload_all(results: dict, round_num: int) -> str:
    """Upload all generated .js files to R2. Returns the CDN base URL for this round."""
    sys.path.insert(0, str(MINER_REF_DIR))
    from miner_reference.generation_pipeline import upload_to_r2

    for stem, js_bytes in results.items():
        url = upload_to_r2(stem, js_bytes, round_id=str(round_num))
        print(f"Uploaded {stem}: {url}", flush=True)

    cdn_url = f"{R2_PUBLIC_URL_BASE}/rounds/{round_num}"
    return cdn_url


def wait_for_reveal_window(round_num: int) -> None:
    """Block until current chain height is within the round's reveal window."""
    schedule_resp = get_round_file(round_num, "schedule.json")
    schedule_resp.raise_for_status()
    schedule = schedule_resp.json()
    earliest = schedule["earliest_reveal_block"]
    latest = schedule["latest_reveal_block"]

    subtensor = bt.subtensor(network=NETWORK)
    current = subtensor.block
    print(f"Reveal window: [{earliest}, {latest}], current block: {current}", flush=True)

    if current < earliest:
        wait_blocks = earliest - current
        print(f"Waiting for {wait_blocks} blocks until reveal window opens...", flush=True)
        subtensor.wait_for_block(earliest)
    elif current > latest:
        raise RuntimeError(
            f"Reveal window already closed (current={current}, latest={latest}). "
            f"Cannot submit for round {round_num}."
        )
    print("Reveal window is open.", flush=True)


def commit_submission(commit_sha: str, cdn_url: str) -> None:
    """Post the on-chain commitment with repo, commit SHA, and CDN URL."""
    wallet = bt.wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY)
    subtensor = bt.subtensor(network=NETWORK)

    payload = json.dumps({
        "repo": GITHUB_REPO,
        "commit": commit_sha,
        "cdn_url": cdn_url,
    })
    print(f"Committing payload: {payload}", flush=True)
    print(f"Wallet: {wallet.hotkey.ss58_address}", flush=True)

    success = subtensor.commit(wallet=wallet, netuid=NETUID, data=payload)
    print(f"Commit result: {success}", flush=True)


def _load_r2_credentials_from_file():
    """Load R2 credentials directly from a fixed file path, bypassing
    shell/tmux environment inheritance entirely. This exists because
    relying on .bashrc + tmux session startup proved unreliable in
    practice -- it silently failed for BOTH round 30 and round 31,
    even when .bashrc was edited well before the session was created.
    """
    import os
    cred_path = os.path.expanduser("~/.r2_credentials")
    if not os.path.exists(cred_path):
        return
    with open(cred_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip()


def main():
    import os
    _load_r2_credentials_from_file()
    # Fail fast, before polling/generating anything, if R2 credentials
    # are missing. This exact gap cost rounds 30 AND 31 -- files were
    # generated successfully but upload crashed with a KeyError, either
    # losing the work entirely (round 30, before the local-save fix) or
    # leaving it stranded on disk after the reveal window closed (round 31).
    missing = [v for v in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not os.environ.get(v)]
    if missing:
        print(f"FATAL: Missing required environment variable(s): {missing}", flush=True)
        print("Set them before running, e.g.:", flush=True)
        print('  export R2_ACCESS_KEY_ID="..."', flush=True)
        print('  export R2_SECRET_ACCESS_KEY="..."', flush=True)
        raise SystemExit(1)
    print("R2 credentials verified present.", flush=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    if args.skip_wait:
        state = get_state()
        round_num = state["current_round"]
        print("Skipping wait, using current round " + str(round_num) + " (stage=" + state["stage"] + ")", flush=True)
        _process_one_round(round_num, args)
        return

    while True:
        try:
            round_num = wait_for_miner_generation_stage()
            _process_one_round(round_num, args)
        except Exception as e:
            print(f"ERROR processing round: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print("Recovering -- resuming poll for the next round in 60s...", flush=True)
            time.sleep(60)


def _process_one_round(round_num, args):
    seed, prompts = fetch_round_data(round_num)

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        print("Limited to first " + str(len(prompts)) + " prompts", flush=True)

    results, failed = generate_all(prompts, seed, round_num)

    if not results:
        print("No successful generations -- aborting, nothing to submit.", flush=True)
        return

    # Save all generated files locally BEFORE attempting upload, so a
    # crash (e.g. missing R2 credentials) never loses completed generation work.
    import os as _os
    local_backup_dir = f"round_{round_num}_output"
    _os.makedirs(local_backup_dir, exist_ok=True)
    for stem, js_bytes in results.items():
        with open(f"{local_backup_dir}/{stem}.js", "wb") as f:
            f.write(js_bytes)
    print(f"Saved {len(results)} files locally to {local_backup_dir}/ before upload attempt", flush=True)

    cdn_url = upload_all(results, round_num)
    commit_sha = get_latest_commit_sha()
    print(f"Repo commit SHA: {commit_sha}", flush=True)

    if args.dry_run:
        print("[DRY RUN] Would commit: repo=" + GITHUB_REPO + " commit=" + commit_sha + " cdn_url=" + cdn_url, flush=True)
        print("[DRY RUN] Skipping reveal wait and on-chain commit.", flush=True)
        return

    wait_for_reveal_window(round_num)
    commit_submission(commit_sha, cdn_url)

    print("Submission complete.", flush=True)


if __name__ == "__main__":
    main()
