from computer_interact.omni_parser2 import OmniParser2
from computer_interact.state.web_agent_state import WebAgentState
import asyncio
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

# Ref 1: https://github.com/ed-donner/llm_engineering/blob/main/week2/day5.ipynb for history inputing
# Ref 2: 


class WebAgent:

    def __init__(self, logger, config):
        self.config = config
        self.logger = logger
        self.omni_parser2 = OmniParser2(logger=self.logger, config=self.config)

    async def setup_browser(self, go_to_page: str):
        print(f"Setting up browser for {go_to_page}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)  # <-- Crucial change
            page = await browser.new_page()
            try:
                await page.goto(go_to_page, timeout=80000, wait_until="domcontentloaded")
            except Exception as e:
                print(f"Error loading page: {e}")
                # Fallback to Google if the original page fails to load/
                
                await page.goto("https://www.google.com", timeout=100000, wait_until="domcontentloaded")

        return browser, page
        
        

    async def run(self, user_query=None):
        #label_coordinates, parsed_content_list = self.omni_parser2.parse()
        browser, page = await self.setup_browser(go_to_page="https://www.google.com")