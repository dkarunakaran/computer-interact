from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
import os
#from langchain_community.agent_toolkits import PlayWrightBrowserToolkit #This is the replaced import
from agent_tools.custom_playwright_toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser


class BrowserAgent:
    def __init__(self, logger, openai_api_token=None):
        self.logger = logger
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_api_token
        self.context = ""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        sync_browser = create_sync_playwright_browser()
        playwright_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
        self.tools = playwright_toolkit.get_tools()
        self.prompt = hub.pull("ebahr/openai-tools-agent-with-context:3b3e6baf") 
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self):
        return self.agent_executor