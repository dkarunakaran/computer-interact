from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import yaml
from web_operator.agents.gmail_agent import GmailAgent
from web_operator.agents.browser_agent import BrowserAgent
from web_operator.agents.agent_state import AgentState
from web_operator.agents.arxiv_agent import ArxivAgent
from typing import Literal, List
from langgraph.graph import StateGraph, START, END
from web_operator.utils import logger_helper
from langgraph.checkpoint.memory import MemorySaver


class Supervisor:
    def __init__(self, required_agents:List[str]):
        self.config = self.__get_config()
        # __ adding infront of the variable and method make them private
        self.__logger = logger_helper(self.config)
        if not os.environ.get("OPENAI_API_KEY"):
            raise KeyError("OPENAI API token is missing, please provide it .env file.") 
        self.required_agents = required_agents
        self.graph_config = {}

    def configure(self):
        self.graph_config = {"configurable": {"thread_id": "1", "recursion_limit": self.config['supervisor']['recursion_limit']}}
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        # Specify the liat of agents
        workers = []
        if 'gmail_agent' in self.required_agents:
            self.__gmail_agent = GmailAgent(cfg=self.config)
            workers.append('gmail_agent')
        if 'arxiv_agent' in self.required_agents:
            self.__arxiv_agent = ArxivAgent(cfg=self.config)
            workers.append('arxiv_agent')
        workers.append('browser_agent')
        self.__browser_agent = BrowserAgent(cfg=self.config)
        workers.append('none')
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: {','.join(workers)}."
            " Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH."
            "If you can't find a suitable worker, then use 'none' worker."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        self.__supervisor_chain = prompt | llm

        # Create langgraph workflow for the agents 
        workflow = StateGraph(AgentState)
        workflow.add_node("Supervisor", self.__supervisor_node)
        workflow.add_node("GmailAgent", self.__gmail_agent_node)
        workflow.add_node("BrowserAgent", self.__browser_agent_node)
        workflow.add_node("ArxivAgent", self.__arxiv_agent_node)
        workflow.add_edge(START, "Supervisor")
        workflow.add_conditional_edges(
            "Supervisor",
            self.__router,
            {"gmail_agent": "GmailAgent", "browser_agent":"BrowserAgent", "arxiv_agent":"ArxivAgent", "__end__": END},
        )
        workflow.add_edge(
            "GmailAgent",
            "Supervisor",
        )
        workflow.add_edge(
            "BrowserAgent",
            "Supervisor",
        )
        workflow.add_edge(
            "ArxivAgent",
            "Supervisor",
        )

        # Set up memory
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)


    def __get_config(self):
        """
        This function defines the config for the library
        """
        cfg = {
            'debug': False,
            'model': 'gpt-4o-mini', #'gpt-4o'
            'GOOGLE_API':{
                'scopes':['https://mail.google.com/']
            },
            'gmail_agent':{
                'recursion_limit': 10,
                'verbose': True,
            },
            'browser_agent': {
                'recursion_limit': 10,
                'verbose': False,
                'headless': True,
            },
            'arxiv_agent': {
                'recursion_limit': 10,
                'verbose': False,
            },
            'supervisor':{
                'recursion_limit': 10
            }
        }

        return cfg


    # This is the router
    def __router(self, state) -> Literal["gmail_agent", "browser_agent", "arxiv_agent", "__end__"]:
            
        # Sleep to avoid hitting QPM limits
        last_result_text = state["supervisor_msg"][-1].content

        if "gmail_agent" in last_result_text:
            return "gmail_agent"
        
        if "browser_agent" in last_result_text:
            return "browser_agent"
        
        if "arxiv_agent" in last_result_text:
            return "arxiv_agent"

        if "none" in last_result_text:
            # Any agent decided the work is done
            return "__end__"
        
        if "FINISH" in last_result_text:
            # Any agent decided the work is done
            return "__end__"
        
        return "Supervisor"

    def __supervisor_node(self, state: AgentState):
        self.__logger.info("Supervisor node started")
        system_message = state["message"][:-1]
        input_message = state["message"][-1]
        result = self.__supervisor_chain.invoke({'system': system_message, 'input': input_message})
        self.__logger.debug(f"Supervisor result:{result}")
        print(result)
        return {'supervisor_msg': [result], 'sender': ['supervisor']}

    def __gmail_agent_node(self, state: AgentState):
        if not 'gmail_agent' in self.required_agents:
            raise KeyError("gmail_agent is not activated, add that to the required_agents varible.") 
        if not os.environ.get("GOOGLE_API_CREDS_LOC"):
            raise KeyError("Local file path of credentials.json is missing, please provide it in .env file.") 
        if not os.environ.get("GOOGLE_API_TOKEN_LOC"):
            raise KeyError("Local file path of token.json is miising, please provide it in .env file.") 
        self.__logger.info("Gmail agent node started")
        context = state["message"][0]
        input = state["message"][-1]
        result = self.__gmail_agent.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": self.__gmail_agent.context+context,"input":input}, {"recursion_limit": self.config['gmail_agent']['recursion_limit']})
        self.__logger.debug(f"Gmail agent result:{result}")
        return {'message': [result['output']], 'sender': ['gmail_agent']}
    
    
    def __browser_agent_node(self, state: AgentState):
        self.__logger.info("Browser agent node started")
        context = state["message"][0]
        input = state["message"][-1]
        result = self.__browser_agent.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": self.__browser_agent.context+context,"input": input}, {"recursion_limit": self.config['browser_agent']['recursion_limit']})
        self.__logger.debug(f"Browser agent result:{result}")
        return {'message': [result['output']], 'sender': ['browser_agent']}
    
    def __arxiv_agent_node(self, state: AgentState):
        self.__logger.info("Arxiv agent node started")
        context = state["message"][0]
        input = state["message"][-1]
        result = self.__arxiv_agent.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": self.__arxiv_agent.context+context,"input":input}, {"recursion_limit": self.config['arxiv_agent']['recursion_limit']})
        self.__logger.debug(f"arxiv agent result:{result}")
        return {'message': [result['output']], 'sender': ['arxiv_agent']}
    
    def run(self, query=None):
        initial_state = AgentState()
        initial_state['message'] = [query]
        result = self.graph.invoke(initial_state, config=self.graph_config)
        self.__logger.info("-------------------------------------")
        self.__logger.info(f"Execution path: {result['sender']}")

    def get_results(self):
        return self.graph.get_state(self.graph_config).values["message"]

        

if __name__ == "__main__":
    load_dotenv()  
    supervisor = Supervisor()
    prompt1 = """
        go to gmail and find email with subject 'Open-Source Rival to OpenAI's Reasoning Model'
        We need only the content of the latest email of the above subject and disgard other emails.
        Extract the first URL (link) from the email content.
        Naviagte to the URL and summarise the content and no further navigation is required

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
    supervisor.run(query=prompt2)
    

   