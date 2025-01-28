from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
import os
#from langchain_community.agent_toolkits import PlayWrightBrowserToolkit #This is the replaced import
from web_operator.agent_tools.custom_playwright_toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser


class BrowserAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        sync_browser = create_sync_playwright_browser()
        playwright_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
        tools = playwright_toolkit.get_tools()
        self.context = """
            You are a browser assistant. you can perform various operation in browser using browser tools.
            Use your tools to answer questions. For example, use fill tool to fill in fields. 
            If you do not have a tool to answer the question, say so.
            If the website is google.com, look for textarea html element instead of input element for filling.
            if the website is duckduckgo.com, look for button element with aria-label with 'Search' for click and input element with id 'searchbox_input' for filling the text.
        """ +  """
                ONLY respond to the part of query relevant to your purpose.
                IGNORE tasks you can't complete. 
        """
        #DO NOT ASK for further assistance unless user specified.
        #DO NOR REPEAT the same search.
        self.prompt = hub.pull("dkarunakaran/openai-tools-agent-with-context") 
        agent = create_openai_tools_agent(llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=self.cfg['BROWSER_AGENT']['verbose'])