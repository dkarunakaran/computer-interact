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
from langchain_core.prompts import ChatPromptTemplate
from agent_state import AgentState
from typing import Literal
from langgraph.graph import StateGraph, START, END

class SupervisorAgents:
    def __init__(self, openai_api_token=None):
        with open("/app/scripts/config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.logger = self.logger_helper(self.cfg)
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_api_token
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        self.gmail_agent = GmailAgent()
        self.browser_agent = BrowserAgent()

        workers_list = ['gmail_operation_agent', 'browser_operation_agent', 'none']

        system_prompt = f"""You are a supervisor tasked with managing a conversation between the
        following workers:  {','.join(workers_list)}. Given the following user request,
        respond with only the worker that needs to act next. Each worker will perform a
        task and respond with their results and status. The produced artifacts need to be of high quality. If you can't find a suitable worker, then use 'none' worker. The produced artifacts need to be of high quality. When finished executing the code,"
    " respond with FINISH. You will NOT perform the the requested taks, but will always use the workers.
        """
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt),("human", "{input}")])
        self.supervisor_chain = prompt | llm
        workflow = StateGraph(AgentState)
        workflow.add_node("Supervisor", self.supervisor_node)
        workflow.add_node("GmailAgent", self.gmail_agent_node)
        workflow.add_node("BrowserAgent", self.browser_agent_node)
        workflow.add_edge(START, "Supervisor")
        workflow.add_conditional_edges(
            "Supervisor",
            self.router,
            {"gmail_operation_agent": "GmailAgent", "browser_operation_agent":"BrowserAgent", "__end__": END},
        )
        workflow.add_edge(
            "GmailAgent",
            "Supervisor",
        )
        workflow.add_edge(
            "BrowserAgent",
            "Supervisor",
        )
        self.graph = workflow.compile()

    # This is the router
    def router(self, state) -> Literal["gmail_operation_agent", "browser_operation_agent", "__end__"]:
            
        # Sleep to avoid hitting QPM limits
        result = state["message"]
        last_result_text = result[-1]

        if "gmail_operation_agent" in last_result_text:
            return "gmail_operation_agent"
        
        if "browser_operation_agent" in last_result_text:
            return "browser_operation_agent"

        if "none" in last_result_text:
            # Any agent decided the work is done
            return "__end__"
        
        if "FINISH" in last_result_text:
            # Any agent decided the work is done
            return "__end__"
        
        return "Supervisor"

    def supervisor_node(self, state: AgentState):
        message = state["message"][-1]
        result = self.supervisor_chain.invoke({'input': message})

        return {'message': [result.content], 'sender': ['supervisor']}

    def gmail_agent_node(self, state: AgentState):
        prompt = state["message"][0]
        result = self.gmail_agent.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": self.browser_agent.context,"input": prompt}, {"recursion_limit": 10})

        return {'message': ["gmail_agent_node run completed"], 'sender': ['gmail_agent']}
    
    def browser_agent_node(self, state: AgentState):
        prompt = state["message"][0]
        result = self.browser_agent.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": self.browser_agent.context,"input": prompt}, {"recursion_limit": 10})
        return {'message': ["browser_agent_node run completed"], 'sender': ['browser_agent']}
        

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
    supervisor = SupervisorAgents(openai_api_token=secret.open_ai_token)
    prompt1 = """
        **Instruction:** 
        go to gmail and find email with subject 'Timesheets to sign for Kids Early Learning Family Day Care'
        We need only the content of the latest email of the above subject and disgard other emails.
        Extract the URL (link) from the email content. 

        **Constraints:**
        - Only extract the first URL found in the email body.
        - If no URL is found, return "No URL found."

        """
    prompt2 = """
            Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and return the summary of results you get. Use fill tool to fill in fields and print out url at each step.
        """
    prompt3 ="""
        do anything
    """
    #output = supervisor.run(prompt2)
    #print(output)

    initial_state = AgentState()
    initial_state['message'] = [prompt1]

    result = supervisor.graph.invoke(initial_state, {"recursion_limit": 10})
    print(result)