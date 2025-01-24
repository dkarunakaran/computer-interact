from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain import hub
import os
import logging
import yaml
from gmail_agent import GmailAgent
from browser_agent import BrowserAgent
from secret import Secret

class SupervisorAgent:
    def __init__(self, openai_api_token=None):
        with open("/app/scripts/config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.logger = self.logger_helper(self.cfg)
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_api_token
        self.context = ""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        self.gmail_agent = GmailAgent(logger=self.logger)
        self.browser_agent = BrowserAgent(logger=self.logger)
        self.tools = [
            Tool(
                name="GmailAgent",
                func=self.gmail_agent.run,
                description="Use this tool for any actions related to Gmail, such as reading emails, sending emails, and managing contacts."
            ),
            Tool(
                name="BrowserAgent",
                func=self.browser_agent.run,
                description="Use this tool for any actions related to web browsing, such as navigating websites, extracting data, and interacting with web elements."
            )
        ]
        self.prompt = hub.pull("ebahr/openai-tools-agent-with-context:3b3e6baf") 
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, prompt):
        return self.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": self.context,"input": prompt})
    
    # Ref - https://medium.com/pythoneers/beyond-print-statements-elevating-debugging-with-python-logging-715b2ae36cd5
    def logger_helper(self, cfg):
        

        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)  # Capture all messages of debug or higher severity

        ### File handler for errors
        # Create a file handler that writes log messages to 'error.log'
        file_handler = logging.FileHandler('error.log') 
        # Set the logging level for this handler to ERROR, which means it will only handle messages of ERROR level or higher
        file_handler.setLevel(logging.ERROR)  

        ### Console handler for info and above
        # Create a console handler that writes log messages to the console
        console_handler = logging.StreamHandler()  
        
        if cfg['debug'] == True:
            console_handler.setLevel(logging.DEBUG)  
        else:
            # Set the logging level for this handler to INFO, which means it will handle messages of INFO level or higher
            console_handler.setLevel(logging.INFO)  

        ### Set formats for handlers
        # Define the format of log messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 
        # Apply the formatter to the file handler 
        file_handler.setFormatter(formatter) 
        # Apply the formatter to the console handler
        console_handler.setFormatter(formatter)  

        ### Add handlers to logger
        # Add the file handler to the logger, so it will write ERROR level messages to 'error.log'
        logger.addHandler(file_handler)  
        # Add the console handler to the logger, so it will write INFO level messages to the console
        logger.addHandler(console_handler)  

        # Now when you log messages, they are directed based on their severity:
        #logger.debug("This will print to console")
        #logger.info("This will also print to console")
        #logger.error("This will print to console and also save to error.log")

        return logger
        

if __name__ == "__main__":
    secret = Secret()
    supervisor = SupervisorAgent(openai_api_token=secret.open_ai_token)
    prompt2 = """
        **Instruction:** 
        go to gmail and find email with subject 'Timesheets to sign for Kids Early Learning Family Day Care'
        We need only the content of the latest email of the above subject and disgard other emails.
        Extract the URL (link) from the email content. 

        **Constraints:**
        - Only extract the first URL found in the email body.
        - If no URL is found, return "No URL found."
        - If URL found, naviagte tot the url and inform us what you see there

        **Output:**
        let me know what action you have done in the URL
        """
    prompt = """
            Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and return the summary of results you get. Use fill tool to fill in fields and print out url at each step.
        """
    output = supervisor.run(prompt2)
    print(output)