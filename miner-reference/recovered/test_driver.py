"""T-driver: unit-test round_driver.do_generate's lock-aware spawn on N6.

Run under .venv-cli (bittensor env; do_generate only needs the stdlib +
subprocess, no torch). Two paths:
  T3a LIVE holder  -> do_generate waits: no spawn, no exception.
  T3b STALE holder -> do_generate SIGKILLs the orphan (single process,
                      never a group kill) and spawns; the spawned runner
                      dies fast (round 994 has no data on GitHub) so a
                      RuntimeError(rc=1) is the EXPECTED outcome.
No production round is touched (fake rounds 994/995).
"""
import json
import os
import subprocess
import sys
import time

MR = "/home/psag-pgx-node6/vidaio-subnet/404-gen-subnet/miner-reference"
sys.path.insert(0, MR)
import round_driver as rd  # noqa: E402

rd.log = lambda m: print("[test-driver] " + m, flush=True)

# T3a: live holder with fresh heartbeat -> wait, do not spawn
sp = subprocess.Popen(["sleep", "300"])
try:
    lp = rd.LOGS / ".gen_lock_995"
    lp.write_text(json.dumps({"pid": sp.pid, "last_heartbeat": "now"}) + "\n")
    time.sleep(0.3)
    rd.do_generate(995)  # must return quietly
    assert not (rd.LOGS / "gen_round_995.log").exists(), "T3a: must NOT spawn while live holder"
    print("T3a driver wait-on-live-lock: PASS", flush=True)
finally:
    sp.terminate()
    sp.wait()

# T3b: live pid but stale heartbeat (3h) -> kill the orphan, spawn fresh
sp2 = subprocess.Popen(["sleep", "300"])
lp = rd.LOGS / ".gen_lock_994"
lp.write_text(json.dumps({"pid": sp2.pid, "last_heartbeat": "2026-09-07T00:00:00Z"}) + "\n")
subprocess.run(["touch", "-d", "3 hours ago", str(lp)])
raised = None
t0 = time.time()
try:
    rd.do_generate(994, deadline_block=9020000, block=9010000)
except RuntimeError as e:
    raised = e
time.sleep(1)
assert sp2.poll() is not None, "T3b: stale orphan must have been SIGKILLed"
assert raised is not None, "T3b: spawned runner for fake round must fail (rc!=0)"
assert "rc=1" in str(raised), "T3b: expected fetch-failure rc=1, got: %s" % raised
print("T3b driver stale-kill + respawn: PASS (%.1fs) - %s" % (time.time() - t0, raised), flush=True)

# leave .gen_lock_994 / gen_round_994.log for the cleanup step to inspect
print("T-DRIVER: ALL PASS", flush=True)
