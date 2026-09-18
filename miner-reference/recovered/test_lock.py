"""T-lock: unit-test gen_runner.acquire_lock on N6 (run under .venv-vllm).

No model load, no GitHub, no R2 - pure lock semantics:
  T1a fresh round  -> acquire True
  T1b live holder  -> acquire False (the r42 duplicate-VLM fix)
  T1c stale lock   -> acquire True  (takeover after a crash)
Cleans up every lock it creates.
"""
import json
import os
import subprocess
import sys
import time

MR = "/home/psag-pgx-node6/vidaio-subnet/404-gen-subnet/miner-reference"
sys.path.insert(0, MR)
import gen_runner  # noqa: E402


def cleanup(*rounds):
    for n in rounds:
        try:
            gen_runner.lock_path(n).unlink()
        except FileNotFoundError:
            pass


# T1a: no lock -> acquire
assert gen_runner.acquire_lock(990) is True, "T1a: fresh lock should be acquired"
print("T1a fresh acquire: PASS", flush=True)

# T1b: live holder (real pid, fresh mtime) -> refuse
sp = subprocess.Popen(["sleep", "300"])
try:
    lp = gen_runner.lock_path(991)
    lp.write_text(json.dumps({"pid": sp.pid, "last_heartbeat": "now"}) + "\n")
    time.sleep(0.3)  # let the mtime settle
    assert gen_runner.acquire_lock(991) is False, "T1b: live holder must be refused"
    print("T1b live holder refused: PASS", flush=True)
finally:
    sp.terminate()
    sp.wait()

# T1c: stale lock (dead pid, old mtime) -> take over.
# pid is one we just spawned and reaped, so it is definitely dead right now.
q = subprocess.Popen(["true"])
q.wait()
lp = gen_runner.lock_path(992)
lp.write_text(json.dumps({"pid": q.pid, "last_heartbeat": "2026-09-07T00:00:00Z"}) + "\n")
subprocess.run(["touch", "-d", "2 hours ago", str(lp)])
assert gen_runner.acquire_lock(992) is True, "T1c: stale lock should be taken over"
print("T1c stale takeover: PASS", flush=True)

cleanup(990, 991, 992)
print("T-LOCK: ALL PASS", flush=True)
