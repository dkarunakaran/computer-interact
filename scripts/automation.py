from playwright.sync_api import sync_playwright
from googleapiclient.discovery import build
import yaml

#from langchain_ollama import ChatOllama
#from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import os
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",      },
)
#from langchain_community.agent_toolkits import PlayWrightBrowserToolkit #This is the replaced import
from tools.custom_playwright_toolkit import PlayWrightBrowserToolkit
from tools.custom_gmail_toolkit import GmailToolkit

import sys
parent_dir = ".."
sys.path.append(parent_dir)
import secret


# Ref 1: https://medium.com/@abhyankarharshal22/mastering-browser-automation-with-langchain-agent-and-playwright-tools-c70f38fddaa6
# Ref 2: 
class Automation:
    def __init__(self):

        with open("config.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.secret = secret.Secret()

        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.secret.open_ai_token

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        #self.model = ChatOllama(model=self.cfg['invoice_agent']['model_for_API_tool'], base_url=self.cfg['invoice_agent']['host'])
        
        sync_browser = create_sync_playwright_browser()
        playwright_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
        tools = playwright_toolkit.get_tools()

        # If we are getting runnable browser error occured as part of the autntication of Google API
        # We need to run get_manual_auth.py outsider of docker to get the token.json to authenticate with GoogleAPI
        gmail_toolkit = GmailToolkit()
        tools.extend(gmail_toolkit.get_tools())
        self.prompt = hub.pull("ebahr/openai-tools-agent-with-context:3b3e6baf") 
        agent = create_openai_tools_agent(self.llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def test(self):
        prompt = """
            Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and return the summary of results you get. Use fill tool to fill in fields and print out url at each step.
        """

        input_msg = """
        go to gmail and get the html content for the email with subject 'Timesheets to sign for Kids Early Learning Family Day Care'.As a next step, get the link from the content and display here.
        """

        prompt2 = """
        **Instruction:** 
        go to gmail and find email with subject 'Timesheets to sign for Kids Early Learning Family Day Care'
        Extract the URL (link) from the email content. 

        **Constraints:**
        - Only extract the first URL found in the email body.
        - If no URL is found, return "No URL found."
        - If URL found, naviagte tot the url and inform us what you see there

        **Output:**
        let me know what action you have done in the URL
        """
        
        context = "If the website is google search, look for textarea html element instead of input element for filling."
        self.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": context,"input": prompt})
       

if __name__ == "__main__":
    automate = Automation()
    automate.test()