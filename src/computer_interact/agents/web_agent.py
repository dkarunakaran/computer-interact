from computer_interact.omni_parser2 import OmniParser2
from computer_interact.state.web_agent_state import WebAgentState
import asyncio
from computer_interact.browser import Browser
import time
from typing import Literal, List
from langgraph.graph import StateGraph, START, END
from computer_interact.state.web_agent_state import AgentState

# Ref 1: https://github.com/ed-donner/llm_engineering/blob/main/week2/day5.ipynb for history inputing
# Ref 2: 


class WebAgent:

    def __init__(self, logger, config):
        self.config = config
        self.logger = logger
        self.omni_parser2 = OmniParser2(logger=self.logger, config=self.config)
        self.browser = Browser()


    async def run(self, user_query=None):
        await self.browser.start()
        page = await self.browser.get_page()
        await page.goto("https://google.com")
        time.sleep(2)
        label_coordinates, parsed_content_list = self.omni_parser2.parse()

    
        
    