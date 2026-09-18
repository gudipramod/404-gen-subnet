"""T-incremental: prove a SIGKILLed gen leaves a REAL partial round on R2.

Patches miner_reference.generation_pipeline so load_models /
generate_scene_for_prompt are fakes (~4s each, no GPU, no VLM) while
upload_to_r2 stays the REAL boto3 path with real creds. Then drives
gen_runner.run() with 8 fake prompts under a fake round 999. The test
supervisor SIGKILLs this process mid-run; afterwards R2 rounds/999/ must
contain the prompts that finished before the kill - that is the crash
rescue: empty-content (r42) becomes partial-content.

Caveat: gen_runner.main() is NOT used (it would fetch round 999 from
GitHub); run() is driven directly, so the driver_uploaded_999 marker is
never written by this test - cleanup is manual (see test plan).
"""
import sys
import time

MR = "/home/psag-pgx-node6/vidaio-subnet/404-gen-subnet/miner-reference"
sys.path.insert(0, MR)
import miner_reference.generation_pipeline as gp  # noqa: E402
import gen_runner  # noqa: E402
import orchestrate_round as orc  # noqa: E402

# main() (bypassed here) loads R2 creds from ~/.r2_credentials into the env
orc._load_r2_credentials_from_file()

N = 999


def fake_gen(image_url, seed):
    time.sleep(4)
    return ("// sn17 selftest fake %s\nconsole.log('ok');\n" % image_url).encode()


gp.load_models = lambda: print("[test] fake load_models", flush=True)
gp.generate_scene_for_prompt = fake_gen
# upload_to_r2: real, untouched

assert gen_runner.acquire_lock(N), "test lock must be acquired"
print("[test] pid=%d starting 8 fake prompts, ~4s each (~35s total + imports)"
      % __import__("os").getpid(), flush=True)
prompts = [{"stem": "fake_%02d" % i, "image_url": "http://fake/%d.png" % i}
           for i in range(8)]
results, failed = gen_runner.run(N, "testseed", prompts)
print("[test] completed all 8: results=%d failed=%d" % (len(results), len(failed)),
      flush=True)
