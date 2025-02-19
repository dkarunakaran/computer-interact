import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # <-- Crucial change
        page = await browser.new_page()
        await page.goto("https://www.example.com")
        await asyncio.sleep(5)  # Keep the browser open for a bit
        await browser.close()

asyncio.run(main())