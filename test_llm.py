import asyncio
import httpx
import time

async def test():
    t0 = time.time()
    try:
        print('Sending request to LLaMA server...')
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post('http://localhost:8080/v1/chat/completions', json={'model': 'qwen', 'messages': [{'role': 'user', 'content': 'hi'}]})
            print('Status:', resp.status_code)
            print('Response:', resp.text)
    except Exception as e:
        print('Error:', type(e).__name__, e)
    print(f'Time taken: {time.time()-t0:.2f}s')

asyncio.run(test())
