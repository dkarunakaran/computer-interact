from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain import hub
import os
import logging
import yaml
from agents.gmail_agent import GmailAgent
from agents.browser_agent import BrowserAgent
from secret import Secret
from langchain_core.prompts import ChatPromptTemplate
from agents.agent_state import AgentState
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from utils import logger_helper

class SupervisorAgents:
    def __init__(self, openai_api_token=None):
        with open("config.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # __ adding infront of the variable and method make them private
        self.__logger = logger_helper(self.config)
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_api_token
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        self.__gmail_agent = GmailAgent(cfg=self.config)
        self.__browser_agent = BrowserAgent(cfg=self.config)
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: 'gmail_operation_agent', 'browser_operation_agent', 'none'."
            " Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH."
            "If you can't find a suitable worker, then use 'none' worker."
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt),("human", "{input}")])
        self.__supervisor_chain = prompt | llm
        workflow = StateGraph(AgentState)
        workflow.add_node("Supervisor", self.__supervisor_node)
        workflow.add_node("GmailAgent", self.__gmail_agent_node)
        workflow.add_node("BrowserAgent", self.__browser_agent_node)
        workflow.add_edge(START, "Supervisor")
        workflow.add_conditional_edges(
            "Supervisor",
            self.__router,
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
    def __router(self, state) -> Literal["gmail_operation_agent", "browser_operation_agent", "__end__"]:
            
        # Sleep to avoid hitting QPM limits
        last_result_text = state["message"][-1].content

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

    def __supervisor_node(self, state: AgentState):
        self.__logger.info("Supervisor node started")
        message = state["message"]
        result = self.__supervisor_chain.invoke(message)
        self.__logger.debug(f"Supervisor result:{result}")
        return {'message': [result], 'sender': ['supervisor']}

    def __gmail_agent_node(self, state: AgentState):
        self.__logger.info("Gmail agent node started")
        context = state["message"][0]
        input = state["message"][-2]
        result = self.__gmail_agent.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": self.__gmail_agent.context+context,"input":input}, {"recursion_limit": self.config['GMAIL_AGENT']['recursion_limit']})
        self.__logger.debug(f"Gmail agent result:{result}")
        return {'message': [result['output']], 'sender': ['gmail_agent']}
    
    def __browser_agent_node(self, state: AgentState):
        self.__logger.info("Browser agent node started")
        context = state["message"][0]
        input = state["message"][-2]
        result = self.__browser_agent.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": self.__browser_agent.context+context,"input": input}, {"recursion_limit": self.config['BROWSER_AGENT']['recursion_limit']})
        self.__logger.debug(f"Browser agent result:{result}")
        return {'message': [result['output']], 'sender': ['browser_agent']}
        

if __name__ == "__main__":
    secret = Secret()
    supervisor = SupervisorAgents(openai_api_token=secret.open_ai_token)
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
    #output = supervisor.run(prompt2)
    #print(output)

    initial_state = AgentState()
    initial_state['message'] = [prompt1]

    result = supervisor.graph.invoke(initial_state, {"recursion_limit": supervisor.config['SUPERVISOR']['recursion_limit']})
    print("-------------------------------------")
    print(f"Execution path: {result['sender']}")

   