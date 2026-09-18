"""Round driver for 404-GEN SN17 (v2 — official bittensor 10.4.0 commit path).

Runs under .venv-cli (bittensor==10.4.0 + bittensor-wallet==4.1.0, the exact
pin of the official 404-Repo/404-cli).

What it does:
  - commits {repo, commit, cdn_url} on-chain for the round whose reveal
    window is currently open: round N while its own window is open, round N+1
    as soon as round N's latest_reveal_block has passed (same round_to_commit
    rule as the official 404-cli)
  - during stage 'miner_generation' of round N: fetches seed/prompts,
    generates all 128 .js files, saves locally, uploads to R2
  - at startup: warns if the coldkey conviction lock on netuid 17 is below
    100 alpha (required by 404-cli, else submissions may not be scored)

Commit API (verified against 404-Repo/404-cli commit.py, bittensor 10.4.0):
  wallet = Wallet(name=..., hotkey=...)   # from bittensor_wallet (4.1.0)
  async with bt.AsyncSubtensor("finney") as s:
      resp = await s.set_reveal_commitment(
          wallet=wallet, netuid=17, data=payload, blocks_until_reveal=2)
  -> resp is an ExtrinsicResponse (.success is the confirmation), and the
     commitment auto-reveals on-chain 2 blocks later in
     Commitments.RevealedCommitments.

Idempotency markers under ../logs/: driver_committed_<round>,
driver_uploaded_<round>.  A restart can never double-commit harmfully
(latest block wins) and never double-uploads (same R2 keys).

v3 reliability changes (2026-09-07, r42 + 09-07 crash-loop postmortem):
  - do_generate is LOCK-AWARE: it will never spawn a second generation
    while a live one holds logs/.gen_lock_<n> (r42 ran duplicate ~16GB
    VLMs because every crash/restart spawned a new gen over orphans).
    Stale orphans (dead pid or heartbeat > GEN_LOCK_MAX_AGE_SEC) are
    killed first; the kill is single-process, never a group kill, so an
    orphan launched by OLD code (which shares the driver's group) cannot
    take the driver down with it.
  - The generation child runs in its OWN session (start_new_session=True);
    a 6h timeout now kills its process group and can never reach the
    driver's.
  - The state log + spawn log report generation_deadline_block when the
    schedule is published, so a missed-deadline incident is visible in
    driver.log instead of being discovered afterwards.
"""
import asyncio
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import bittensor as bt
from bittensor_wallet import Wallet  # 10.4.0 has no bt.wallet factory

import orchestrate_round as orc

BASE = Path(__file__).resolve().parent      # miner-reference/
LOGS = BASE.parent / "logs"                 # 404-gen-subnet/logs/
LOG_FILE = LOGS / "driver.log"
PID_FILE = LOGS / ".driver_pid"
ACTIVE_COMP = "https://raw.githubusercontent.com/404-Repo/404-active-competition/main"

TORCH_VENV_PY = str(BASE / ".venv-miner" / "bin" / "python")
GEN_TIMEOUT_SECONDS = 6 * 3600
GEN_LOCK_MAX_AGE_SEC = 1800    # must match gen_runner.LOCK_MAX_AGE_SEC


def gen_lock_state(n):
    """Return (path, holder_pid, alive, age_sec) for the round's gen lock.

    The lock is written and heartbeated by gen_runner.py (see its header:
    one live generation per round). age_sec is None when no lock exists.
    """
    lp = LOGS / (".gen_lock_%d" % n)
    if not lp.exists():
        return lp, None, False, None
    holder = None
    try:
        holder = int(json.loads(lp.read_text()).get("pid"))
    except Exception:
        pass
    age = int(time.time() - lp.stat().st_mtime)
    alive = False
    if holder:
        try:
            os.kill(holder, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
    return lp, holder, alive, age


def log(msg):
    line = "[driver] " + msg
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def marker(kind, n):
    return LOGS / ("driver_%s_%d" % (kind, n))


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_schedule(n):
    try:
        return fetch_json(ACTIVE_COMP + "/rounds/%d/schedule.json" % n)
    except Exception:
        return None


def current_block():
    # bittensor 10.4.0 has no bt.subtensor() factory; AsyncSubtensor.block
    # is a coroutine property (verified by probe: await s.block -> int).
    async def _b():
        async with bt.AsyncSubtensor(orc.NETWORK) as s:
            return await s.block

    return asyncio.run(_b())


def window_for(cur, cur_sched, block):
    """Return (target_round, {'earliest_reveal_block','latest_reveal_block'}).

    Official 404-cli rule: commits land in round N while the chain is at or
    before round N's latest_reveal_block; after that, they belong to N+1.
    """
    latest = cur_sched.get("latest_reveal_block", 2 ** 62)
    if block <= latest:
        return cur, cur_sched
    target = cur + 1
    ns = fetch_schedule(target)
    if ns:
        return target, ns
    earliest = cur_sched.get("earliest_reveal_block")
    derived = {
        "earliest_reveal_block": latest + 2,
        "latest_reveal_block": latest + (latest - earliest),
    }
    log("schedule for round %d not published yet; using derived window %s" % (target, derived))
    return target, derived


def lock_alpha(lock):
    """Alpha (TAO) conviction-locked for the coldkey on netuid 17; 0.0 if none.

    bittensor 10.4.0's get_coldkey_lock returns a LockState object (fields:
    locked_mass (Balance), conviction, last_update), not a dict — handle
    both shapes in case the pin changes.
    """
    if lock is None:
        return 0.0
    if isinstance(lock, dict):
        m = lock.get("locked_mass")
    else:
        m = getattr(lock, "locked_mass", None)
    if m is None:
        return 0.0
    try:
        return float(m.tao)  # conviction locks are 1:1 alpha=TAO
    except AttributeError:
        return int(m.rao) / 1e6


def do_commit(n, block):
    sha = orc.get_latest_commit_sha()
    cdn_url = orc.R2_PUBLIC_URL_BASE + "/rounds/" + str(n)
    payload = json.dumps({"repo": orc.GITHUB_REPO, "commit": sha, "cdn_url": cdn_url})
    wallet = Wallet(name=orc.WALLET_NAME, hotkey=orc.WALLET_HOTKEY)

    async def _c():
        async with bt.AsyncSubtensor(orc.NETWORK) as s:
            return await s.set_reveal_commitment(
                wallet=wallet, netuid=orc.NETUID, data=payload, blocks_until_reveal=2)

    log("round %d: committing on-chain at block=%d payload=%s wallet=%s"
        % (n, block, payload, wallet.hotkey.ss58_address))
    resp = asyncio.run(_c())
    # bittensor 10.4.0: set_reveal_commitment returns an ExtrinsicResponse
    # (fields incl. .success), NOT a (success, block) tuple.
    log("round %d: set_reveal_commitment response=%s" % (n, resp))
    ok = getattr(resp, "success", None)
    if isinstance(resp, tuple):
        ok = resp[0]
    if ok:
        marker("committed", n).write_text(sha + "\n")
        log("round %d: commit marker written (auto-reveals ~2 blocks later)" % n)
    else:
        # Never mark a non-confirmed commit: re-committing is safe
        # (latest block wins), so retrying next poll is the correct move.
        raise RuntimeError("on-chain commit for round %d not confirmed: %s" % (n, resp))


def do_generate(n, deadline_block=None, block=None):
    # Generation needs torch + transformers (local VLM + validator loop),
    # which .venv-cli (the commit/monitor venv) lacks. Round 39 died here
    # in-process with "No module named 'torch'"; shell out to gen_runner.py
    # under .venv-vllm instead. The runner saves + uploads per prompt
    # (r42 postmortem: all-at-end upload lost the round when the process
    # died mid-generation) and writes the driver_uploaded_<n> marker itself.
    #
    # LOCK-AWARE (r42 postmortem): r42's crash storm restarted this driver
    # 9 times; every restart spawned a NEW generation while orphaned old
    # ones kept running, so two ~16GB VLMs sat on the GPU at once and the
    # NVRM OOM storm compounded itself. Never spawn a duplicate:
    #   - live holder  -> wait; the running gen is the asset, and when it
    #     finishes its marker will make the main loop skip this entirely.
    #   - stale holder -> it's an orphan holding GPU for nothing: kill the
    #     process (single process, not its group - an orphan from OLD code
    #     shares the driver's process group, and killing that group would
    #     kill the driver too) and spawn fresh.
    lp, holder, alive, age = gen_lock_state(n)
    if alive and age < GEN_LOCK_MAX_AGE_SEC:
        log("round %d: generation already in progress (pid %d, heartbeat %ds ago) - waiting, NOT spawning a duplicate"
            % (n, holder, age))
        return
    if alive and age >= GEN_LOCK_MAX_AGE_SEC:
        log("round %d: killing stale generation orphan pid %d (heartbeat %ds old > %ds)"
            % (n, holder, age, GEN_LOCK_MAX_AGE_SEC))
        try:
            os.kill(holder, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    dl_msg = ""
    if deadline_block and block:
        left = max(0, deadline_block - block)
        dl_msg = " (generation deadline block %s: %d blocks / ~%dh left)" % (
            deadline_block, left, left * 12 // 3600)
    gen_log = LOGS / ("gen_round_%d.log" % n)
    log("round %d: generation starting under %s (log: %s)%s" % (n, TORCH_VENV_PY, gen_log, dl_msg))
    with open(str(gen_log), "a") as f:
        f.write("[gen] round %d started %s\n" % (n, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        f.flush()
        # start_new_session: gen_runner leads its OWN process group so a
        # timeout killpg can never reach this driver's group.
        p = subprocess.Popen([TORCH_VENV_PY, str(BASE / "gen_runner.py"), str(n)],
                             cwd=str(BASE), stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=GEN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            log("round %d: generation exceeded %ds - killing process group %d"
                % (n, GEN_TIMEOUT_SECONDS, p.pid))
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                p.kill()
            rc = p.wait()
    if rc != 0:
        # rc=4 means the runner saw another live gen grab the lock between
        # our check and its start; the generic retry-next-poll handles it.
        raise RuntimeError("round %d: generation runner failed rc=%d (see %s)"
                           % (n, rc, gen_log))
    log("round %d: generation + upload finished" % n)


def check_lock():
    try:
        wallet = Wallet(name=orc.WALLET_NAME, hotkey=orc.WALLET_HOTKEY)
        ck = wallet.coldkeypub.ss58_address

        async def _l():
            async with bt.AsyncSubtensor(orc.NETWORK) as s:
                return await s.get_coldkey_lock(coldkey_ss58=ck, netuid=orc.NETUID)

        lock = asyncio.run(_l())
        alpha = lock_alpha(lock)
        log("conviction lock on netuid %d: %.1f alpha (required >= 100); raw=%s"
            % (orc.NETUID, alpha, lock))
        if alpha < 100.0:
            log("WARNING: conviction lock below 100 alpha - submissions may not be scored")
    except Exception as e:
        log("lock check failed: %s" % e)


def main():
    LOGS.mkdir(exist_ok=True)
    PID_FILE.write_text(str(os.getpid()) + "\n")
    log("driver v2 started pid=%d (bittensor %s)" % (os.getpid(), getattr(bt, "__version__", "?")))
    orc._load_r2_credentials_from_file()
    log("R2 credentials loaded")
    check_lock()
    last_seen = None
    last_win = None
    while True:
        try:
            state = orc.get_state()
            cur = int(state["current_round"])
            stage = state["stage"]
            block = current_block()
            sched = fetch_schedule(cur)
            key = (cur, stage)
            if key != last_seen:
                dl_info = ""
                dl = sched.get("generation_deadline_block") if sched else None
                if dl and stage == "miner_generation":
                    left = max(0, int(dl) - block)
                    dl_info = " gen_deadline=%s (%d blocks / ~%dh left)" % (dl, left, left * 12 // 3600)
                log("state: round=%d stage=%s next_stage_eta=%s block=%d%s"
                    % (cur, stage, state.get("next_stage_eta"), block, dl_info))
                last_seen = key
            if sched:
                target, win = window_for(cur, sched, block)
                in_win = win["earliest_reveal_block"] <= block <= win["latest_reveal_block"]
                winkey = (target, in_win, win["latest_reveal_block"])
                if winkey != last_win:
                    log("window: target_round=%d in_window=%s block=%d window=[%s, %s]"
                        % (target, in_win, block, win["earliest_reveal_block"],
                           win["latest_reveal_block"]))
                    last_win = winkey
                if in_win and not marker("committed", target).exists():
                    try:
                        do_commit(target, block)
                    except Exception as e:
                        log("round %d: commit failed: %s (retrying next poll)" % (target, e))
            if stage == "miner_generation" and not marker("uploaded", cur).exists():
                dl = sched.get("generation_deadline_block") if sched else None
                try:
                    do_generate(cur, deadline_block=dl, block=block)
                except Exception as e:
                    log("round %d: generation/upload failed: %s" % (cur, e))
        except Exception as e:
            log("poll error: %s" % e)
        time.sleep(orc.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
