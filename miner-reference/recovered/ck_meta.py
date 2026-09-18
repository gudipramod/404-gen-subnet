import json
p="/home/psag-pgx-node6/.bittensor/wallets/my_chrome_wallet/coldkey"
data=open(p).read()
head=data.split("\n",1)[0]
try:
    meta=json.loads(head)
    print("CKADDR:", meta.get("address"))
except Exception as e:
    # maybe whole file is one json
    try:
        meta=json.loads(data)
        print("CKADDR(full):", meta.get("address"))
    except Exception as e2:
        print("meta parse fail:", type(e).__name__, str(e)[:80], "|", type(e2).__name__, str(e2)[:80])
