import asyncio
import sys
# append parent dir to sys.path if needed
sys.path.append('C:\\Users\\david\\.gemini\\antigravity\\scratch\\erisv4')

from eris.interface.mediator import OpenAIBackend

async def test():
    backend = OpenAIBackend(model='qwen', api_key='sk-no-key', base_url='http://localhost:8080/v1')
    try:
        res = await backend.generate(prompt='hi', system='you are eris', max_tokens=2000)
        print('Raw:', res.raw)
        print('Success!', repr(res.text[:50]))
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
