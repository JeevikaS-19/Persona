import sys
import logging
import asyncio
import traceback

logging.basicConfig(level=logging.DEBUG)

import main

async def test():
    try:
        res = await main.analyze_video_production(r'D:\real video\real1.mp4', 'upload')
        print("\n\nFINAL RESULT:")
        print(res)
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
