from playwright.sync_api import sync_playwright
from googleapiclient.discovery import build
import yaml
import re
import base64
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
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
import utility
import secret
import google_ai_studio_services

# Ref 1: https://medium.com/@abhyankarharshal22/mastering-browser-automation-with-langchain-agent-and-playwright-tools-c70f38fddaa6
# Ref 2: 
class Automation:
    def __init__(self):
        with open("/app/config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.logger = utility.logger_helper()

        self.secret = secret.Secret()

        # Authenticate with GOOGLE API
        creds = utility.authenticate(self.cfg)
        self.gmail_service = build("gmail", "v1", credentials=creds)

        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.secret.open_ai_token

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        #self.model = ChatOllama(model=self.cfg['invoice_agent']['model_for_API_tool'], base_url=self.cfg['invoice_agent']['host'])
        
        sync_browser = create_sync_playwright_browser()
        playwright_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
        tools = playwright_toolkit.get_tools()
        gmail_toolkit = GmailToolkit()
        tools.extend(gmail_toolkit.get_tools())
        self.prompt = hub.pull("ebahr/openai-tools-agent-with-context:3b3e6baf") 
        agent = create_openai_tools_agent(self.llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def test(self):
        input_msg = """
            Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and return the summary of results you get. Use fill tool to fill in fields and print out url at each step.
        """

        input_msg = """
        go to gmail and get the html content for the email with subject 'Timesheets to sign for Kids Early Learning Family Day Care'.As a next step, get the link from the content and display here.
        """

        prompt = """
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
       
    
    def get_emails(self):
        result = self.gmail_service.users().messages().list(maxResults=self.cfg['GOOGLE_API']['no_emails'], userId='me').execute()
        messages = result.get('messages')
        self.logger.info("Got the mails and processing now")
        # messages is a list of dictionaries where each dictionary contains a message id.
        # iterate through all the messages
        for msg in messages:
            # Get the message from its id
            txt = self.gmail_service.users().messages().get(userId='me', id=msg['id']).execute()
            
            # Get value of 'payload' from dictionary 'txt'
            payload = txt['payload']
            headers = payload['headers']
            subject = ""

            # Look for Subject and Sender Email in the headers
            for d in headers:
                if d['name'] == 'Subject':
                    subject = d['value']
                if d['name'] == 'From':
                    match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', d['value'])
                    sender = match.group(0)

            # Code for getting only email we want to track
            proceed = False
            subjects = ['Timesheets to sign for Kids Early Learning Family Day Care']
            #subjects = ['Open-Source Rival to OpenAI']
            subject_found = [True if s in subject else False  for s in subjects]
            if True in subject_found:
                proceed = True

            if proceed:
                self.logger.info(f"Processing: '{subject}' started")
                print(payload)
                data = payload['parts'][0]['body']['data']
                data = data.replace("-","+").replace("_","/")
                decoded_data = base64.b64decode(data)
                soup = BeautifulSoup(decoded_data , "lxml")
                html_content = str(soup)            
                #prompt1 = "Extrach the link from this text:"
                #prompt2 = "" #" Only give the link and no other text."
                # Query the LLM to extract the link
                #result_with_url = self.gen_ai_google.generate_content(prompt1+html_content+prompt2)
                # Browser tools for langraph: https://python.langchain.com/docs/integrations/tools/playwright/
                #print(result_with_url)

                context = ""
                input_msg = "find the link from the html content and navigate to the site. html content:"+html_content
                self.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": context,"input": input_msg})
                


        
    def browser_run(self):
        with sync_playwright() as p:
            # Channel can be "chrome", "msedge", "chrome-beta", "msedge-beta" or "msedge-dev".
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://playwright.dev")
            print(page.title())
            browser.close()


if __name__ == "__main__":
    automate = Automation()
    automate.test()