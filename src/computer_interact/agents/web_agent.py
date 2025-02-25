from computer_interact.omni_parser2 import OmniParser2
from typing import Literal, List
from langgraph.graph import StateGraph, START, END
from computer_interact.state.web_agent_state import WebAgentState
from computer_interact.tools.computer_use_tools import click
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
import os
import json
import  computer_interact.agents.prompts as prompts 

# Ref 1: https://github.com/ed-donner/llm_engineering/blob/main/week2/day5.ipynb for history inputing
# Ref 2: https://github.com/langchain-ai/ollama-deep-researcher/blob/main/src/assistant/graph.py
# Ref 3: https://github.com/dkarunakaran/computer-interact/blob/e90b33962533f8064f4336181149132fe0d0e250/src/web_operator/nodes/computer_use_node.py
"""
URL decider(LLM) -> open browser and play to url using playwright -> 
omniparser -> nextstep (LLM)-> omniparser -> loop untill LLM stop for the next step

"""


class WebAgent:

    def __init__(self, logger, config):
        self.config = config
        self.logger = logger
        self.omni_parser2 = OmniParser2(logger=self.logger, config=self.config)
        self.graph_config = {"configurable": {"thread_id": "1", "recursion_limit": 20}}
        # Add nodes and edges 
        workflow = StateGraph(WebAgentState)
        workflow.add_node("llm", self.llm_node)
        workflow.add_node("annotate", self.annotate_node)
        workflow.add_node("gui_action", self.gui_action_node)
        """workflow.add_node("finalize_summary", self.finalize_summary_node)"""

        # Add edges
        workflow.add_edge(START, "annotate")
        workflow.add_edge("annotate", "llm")
        workflow.add_conditional_edges(
            "llm", 
            self.router
        )
        workflow.add_edge(
            "gui_action",
            "annotate"
        )
        """
        workflow.add_edge("finalize_summary", END)"""

        # Set up memory
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)


    # Define the function that determines whether to continue or not
    def router(self, state: WebAgentState) -> Literal["annotate", "gui_action", END]:

        last_action = state['action'][-1]
        steps = json.loads(last_action)
        # If the LLM makes a tool call, then we route to the "tools" node
        if "annotate" in steps['node']:
            return "annotate"
        
        if "gui_action" in steps['node']:
            return "gui_action"
        
        # Otherwise, we stop (reply to the user)
        return END

    def annotate_node(self, state:WebAgentState):

        self.logger.info("annotate node started...")
        label_coordinates, parsed_content_list = self.omni_parser2.parse()

        return {'parsed_content_list': str(parsed_content_list), 'sender': ['annotate']}
        

    def gui_action_node(self, state:WebAgentState):
        self.logger.info("gui_action node started...")

        return {'actions_taken': [""], 'sender': ['gui_action']}
            

    def llm_node(self, state:WebAgentState):

        self.logger.info("llm_node started...")
        system_msg = prompts.system_msg_llm_node_web_agent
        user_query = state['user_query']
        parsed_content_list = state['parsed_content_list']
        actions_taken = state['actions_taken']
        llm = OpenAI(
            #api_key=os.environ.get("GEMINI_API_KEY"),
            #base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        messages = [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': f"User query: {user_query}"},
            {'role': 'user', 'content': f"List of icon/text box description: {parsed_content_list}"},
            {'role': 'user', 'content': f"Actions Taken So far: {actions_taken}"},
            {'role': 'user', 'content': f"Urls Already Visited: "}
        ]
        completion = llm.chat.completions.create(
            #model='gemini-2.0-pro-exp-02-05',
            model='gpt-4o-mini',
            messages=messages
        )

        content = completion.choices[0].message.content
        content = content.strip().replace("json", "").replace("", "").strip()
        action = content.strip().replace("```", "").replace("", "").strip()

        print(action)

        return {'action': [action], 'sender': ['llm']}

    def finalize_summary_node(self, state:WebAgentState):
        pass

    def run(self, user_query=None):
        #label_coordinates, parsed_content_list = self.omni_parser2.parse()
        initial_state = WebAgentState()
        initial_state['user_query'] = user_query
        result = self.graph.invoke(initial_state, config=self.graph_config)
        print(result)

    
        
    