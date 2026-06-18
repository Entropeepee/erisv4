import asyncio
import sys
import os

sys.path.append('C:\\Users\\david\\.gemini\\antigravity\\scratch\\erisv4')
os.environ['ERIS_ENV'] = 'dev'

from eris.orchestrator import ErisOrchestrator

async def main():
    orchestrator = ErisOrchestrator()
    print('Calling process()...')
    try:
        res = await orchestrator.process('hello eris are you there?')
        print('Result:', res.response_text)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
