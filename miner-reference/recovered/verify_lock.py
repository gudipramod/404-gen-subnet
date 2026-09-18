import asyncio
import bittensor as bt

CK = "5DUVGKccY9g6Dw4KyDVTmhQgWZqPpcHfy8ttm8E26noyWnmU"
HK = "5HTSaZVvCdRVi4Pbf2WUpU764SBQ1nmQw9dBwAu4Wbn8En7o"
NETUID = 17

async def main():
    async with bt.AsyncSubtensor("finney") as s:
        print("BLOCK:", await s.block)
        cl = await s.get_coldkey_lock(coldkey_ss58=CK, netuid=NETUID)
        print("COLDKEY_LOCK:", cl)
        sl = await s.get_stake_lock(coldkey_ss58=CK, netuid=NETUID, hotkey_ss58=HK)
        print("STAKE_LOCK:", sl)

asyncio.run(main())
