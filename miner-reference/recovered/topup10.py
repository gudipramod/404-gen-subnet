import asyncio
import bittensor as bt
from bittensor_wallet import Wallet

WALLET = "my_chrome_wallet"
HOTKEY = "gen404_hotkey"
CK = "5DUVGKccY9g6Dw4KyDVTmhQgWZqPpcHfy8ttm8E26noyWnmU"
NETUID = 17
AMOUNT = bt.Balance.from_tao(10.0)   # +10 alpha conviction top-up

async def main():
    w = Wallet(name=WALLET, hotkey=HOTKEY)
    async with bt.AsyncSubtensor("finney") as s:
        print("BLOCK:", await s.block)
        hk = w.hotkey.ss58_address
        print("HK:", hk)
        cl = await s.get_coldkey_lock(coldkey_ss58=CK, netuid=NETUID)
        print("COLDKEY_LOCK_BEFORE:", cl)
        stake = await s.get_stake(coldkey_ss58=CK, hotkey_ss58=hk, netuid=NETUID)
        print("STAKE_BEFORE:", stake)
        r = await s.lock_stake(wallet=w, hotkey_ss58=hk, netuid=NETUID, amount=AMOUNT)
        print("LOCK_RESULT:", r)
        await asyncio.sleep(6)
        cl = await s.get_coldkey_lock(coldkey_ss58=CK, netuid=NETUID)
        print("COLDKEY_LOCK_AFTER:", cl)
        sl = await s.get_stake_lock(coldkey_ss58=CK, netuid=NETUID, hotkey_ss58=hk)
        print("STAKE_LOCK_AFTER:", sl)

asyncio.run(main())
