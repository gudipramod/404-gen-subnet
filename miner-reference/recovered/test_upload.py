"""T-upload: prove the REAL R2 upload path end-to-end, then clean up.

Uploads one selftest .js under rounds/999/, fetches it back over the public
CDN URL (must match byte-for-byte), deletes the key, and confirms the URL
no longer serves it. No production round is touched.
"""
import sys
import urllib.error
import urllib.request

MR = "/home/psag-pgx-node6/vidaio-subnet/404-gen-subnet/miner-reference"
sys.path.insert(0, MR)
import orchestrate_round as orc  # noqa: E402
from miner_reference.generation_pipeline import (  # noqa: E402
    R2_BUCKET, _get_r2_client, upload_to_r2)

orc._load_r2_credentials_from_file()
STEM = "sn17_selftest_0907"
body = b"// sn17 reliability selftest 2026-09-07 - safe to delete\nconsole.log('ok')\n"
url = upload_to_r2(STEM, body, round_id="999")
print("[test] uploaded:", url, flush=True)
data = urllib.request.urlopen(url, timeout=30).read()
assert data == body, "T-upload: public fetch mismatch"
print("T-upload public fetch: PASS (%d bytes)" % len(data), flush=True)

client = _get_r2_client()
client.delete_object(Bucket=R2_BUCKET, Key="rounds/999/%s.js" % STEM)
try:
    urllib.request.urlopen(url, timeout=30)
    raise SystemExit("T-upload: key still fetchable after delete")
except urllib.error.HTTPError as e:
    assert e.code in (403, 404), "T-upload: unexpected status %s" % e.code
    print("T-upload delete + gone verify: PASS (HTTP %d)" % e.code, flush=True)
print("T-UPLOAD: ALL PASS", flush=True)
