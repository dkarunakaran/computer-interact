from playwright.async_api import async_playwright
import asyncio
import signal
import sys

class Browser:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.running = True
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        print("\nClosing browser gracefully...")
        self.running = False
        
    async def start(self):
        """Start the browser if it's not already running"""
        if not self.browser:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=False,
                args=['--start-maximized']
            )
            self.context = await self.browser.new_context(
                viewport=None,
                no_viewport=True
            )
    
    async def get_page(self):
        """Get a new page in the existing browser context"""
        if not self.browser:
            await self.start()
        return await self.context.new_page()
    
    async def close(self):
        """Clean up resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
            
    async def keep_alive(self):
        """Keep the browser running indefinitely"""
        try:
            print("Browser is running. Press Ctrl+C to stop...")
            while self.running:
                await asyncio.sleep(1)
        finally:
            await self.close()