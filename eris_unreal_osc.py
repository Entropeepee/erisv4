"""
Eris to Unreal Engine 5 Bridge (OSC)
=====================================

This script connects to Eris's WebSocket server and converts the incoming
vitals, cognitive regime, and TTS audio into OSC (Open Sound Control) messages
that Unreal Engine 5 can ingest natively.

Prerequisites:
    pip install websockets python-osc

Usage:
    1. In Unreal Engine, enable the "OSC" plugin.
    2. Create an OSC Server in Blueprint listening on port 8000.
    3. Run Eris Echo v4 (`python -m eris.server.app`).
    4. Run this script: `python eris_unreal_osc.py`
"""

import asyncio
import json
import websockets
from pythonosc.udp_client import SimpleUDPClient

# Configuration
ERIS_WS_URL = "ws://127.0.0.1:8001/ws"
UNREAL_OSC_IP = "127.0.0.1"
UNREAL_OSC_PORT = 9000

client = SimpleUDPClient(UNREAL_OSC_IP, UNREAL_OSC_PORT)

async def connect_to_eris():
    print(f"Connecting to Eris at {ERIS_WS_URL}...")
    try:
        async with websockets.connect(ERIS_WS_URL) as ws:
            print("Connected! Forwarding vitals to Unreal Engine OSC...")
            while True:
                message = await ws.recv()
                data = json.loads(message)
                
                # Map Eris vitals to OSC addresses for Unreal
                client.send_message("/eris/coherence", data.get("coherence", 0.0))
                client.send_message("/eris/dCdX", data.get("dCdX", 0.0))
                client.send_message("/eris/regime", data.get("regime", "unknown"))
                client.send_message("/eris/archetype", data.get("archetype", "unknown"))
                
                # Use dCdX to drive the MetaHuman's 'Thinking' blendshapes or brow furrow
                dCdX = float(data.get("dCdX", 0.0))
                client.send_message("/metahuman/ctrl_brow_furrow", min(1.0, dCdX * 2.0))
                
                print(f"Forwarded: Archetype={data.get('archetype')} | dCdX={dCdX:.3f}")
                
    except Exception as e:
        print(f"Disconnected or error: {e}")
        print("Retrying in 3 seconds...")
        await asyncio.sleep(3)
        await connect_to_eris()

if __name__ == "__main__":
    asyncio.run(connect_to_eris())
