"""A2: the workload governor keeps foreground chat ahead of background loops."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import unittest

from eris.server.governor import Governor


class TestGovernor(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_is_passthrough(self):
        os.environ["ERIS_GOVERNOR"] = "off"
        try:
            g = Governor()
            order = []
            async with g.foreground():
                order.append("fg")
            async with g.background():
                order.append("bg")
            self.assertEqual(order, ["fg", "bg"])
        finally:
            os.environ.pop("ERIS_GOVERNOR", None)

    async def test_background_defers_to_active_foreground(self):
        g = Governor()
        entered = asyncio.Event()

        async def bg():
            async with g.background():
                entered.set()

        async with g.foreground():
            task = asyncio.create_task(bg())
            await asyncio.sleep(0.05)
            self.assertFalse(entered.is_set(), "background ran during foreground")
        await asyncio.wait_for(task, timeout=1.0)
        self.assertTrue(entered.is_set())          # proceeds once foreground exits

    async def test_background_serializes(self):
        g = Governor()
        active = 0
        peak = 0

        async def bg():
            nonlocal active, peak
            async with g.background():
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.03)
                active -= 1

        await asyncio.gather(*(bg() for _ in range(4)))
        self.assertEqual(peak, 1)                  # never two background at once

    async def test_foreground_never_blocks_on_background(self):
        g = Governor()
        got_fg = asyncio.Event()

        async def bg():
            async with g.background():
                await asyncio.sleep(0.2)           # hold the bg slot a while

        task = asyncio.create_task(bg())
        await asyncio.sleep(0.01)
        # Foreground must enter immediately even while background holds its slot.
        async with g.foreground():
            got_fg.set()
        self.assertTrue(got_fg.is_set())
        await task


if __name__ == "__main__":
    unittest.main()
