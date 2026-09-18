import asyncio, sys
import bittensor as bt
from bittensor_wallet import Wallet

CK = "5DUVGKccY9g6Dw4KyDVTmhQgWZqPpcHfy8ttm8E26noyWnmU"
NETUID = 17
WALLET = "my_chrome_wallet"
HOTKEY = "gen404_hotkey"

async def chain():
    w = Wallet(name=WALLET, hotkey=HOTKEY)
    hk = w.hotkey.ss58_address
    async with bt.AsyncSubtensor("finney") as s:
        print("BLOCK:", await s.block)
        try:
            cl = await s.get_coldkey_lock(coldkey_ss58=CK, netuid=NETUID)
            print("COLDKEY_LOCK:", cl)
        except Exception as e:
            print("COLDKEY_LOCK err:", type(e).__name__, e)
        try:
            stake = await s.get_stake(coldkey_ss58=CK, hotkey_ss58=hk, netuid=NETUID)
            print("STAKE:", stake)
        except Exception as e:
            print("STAKE err:", type(e).__name__, e)

asyncio.run(chain())

# --- read-only creds-file password test (reads pw from stdin, no signing) ---
try:
    w = Wallet(name=WALLET, hotkey=HOTKEY)
    w.unlock_coldkey()  # reads password from stdin
    addr = w.coldkey.ss58_address
    print("CRED_TEST: UNLOCKED_OK CK_MATCH=", (addr == CK))
except Exception as e:
    print("CRED_TEST: FAIL", type(e).__name__, str(e)[:80])
