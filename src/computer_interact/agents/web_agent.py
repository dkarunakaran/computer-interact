from computer_interact.omni_parser2 import OmniParser2
import asyncio
from computer_interact.browser import Browser
import time
from typing import Literal, List
from langgraph.graph import StateGraph, START, END
from computer_interact.state.web_agent_state import WebAgentState
from computer_interact.tools.computer_use_tools import click
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
import os
import json
from computer_interact.agents.web_agent_json_schema import response_json_supervisor, response_json_supervisor_wo_steps_details

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
        self.graph_config = {"configurable": {"thread_id": "1", "recursion_limit": 10}}
        # Add nodes and edges 
        workflow = StateGraph(WebAgentState)
        workflow.add_node("supervisor_llm", self.supervisor_llm_node)
        workflow.add_node("omni_parser", self.omni_parser_node)
        """workflow.add_node("gui_action", self.gui_action_node)
        workflow.add_node("finalize_summary", self.finalize_summary_node)"""

        # Add edges
        workflow.add_edge(START, "supervisor_llm")
        workflow.add_conditional_edges(
            "supervisor_llm", 
            self.router
        )
        workflow.add_edge(
            "omni_parser",
            "supervisor_llm"
        )
        """workflow.add_edge(
            "gui_action",
            "supervisor_llm"
        )
        workflow.add_edge("finalize_summary", END)"""

        # Set up memory
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)


    # Define the function that determines whether to continue or not
    def router(self, state: WebAgentState) -> Literal["omni_parser", END]:
        messages = state['messages']
        last_message = messages[-1]
        steps = json.loads(last_message)
        # If the LLM makes a tool call, then we route to the "tools" node
        if "omni_parser" in steps['next_step']:
            return "omni_parser"
        # Otherwise, we stop (reply to the user)
        return END

    def omni_parser_node(self, state:WebAgentState):

        self.logger.info("omni_parser node started...")
        label_coordinates, parsed_content_list = self.omni_parser2.parse()
        message = [
                {"role": "user", "content": label_coordinates},
                {"role": "user", "content": parsed_content_list},
                {"role": "user", "content": "Here is the details on screen and No supervisor needs to make decision what to do from here"}
            ]
        
        return {'messages': [message], 'sender': ['omni_parser']}
        

    def gui_action_node(self, state:WebAgentState):
        system_msg = """
            Use a mouse and keyboard to interact with a computer, and take screenshots.
            * This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
            * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
            * Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
            * If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
            * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
        """
        tools = [click]
        
        

    def supervisor_llm_node(self, state:WebAgentState):

        self.logger.info("Supervsor node started...")
        system_msg = """
        You are a supervisor and tasked to select the right node for further automation.
        You have three nodes: omni_parser, gui_action, and finalize_summary. 
        Yor task is to select the right node based on the automation history.
        you start with creating plan for the web automation based on the user query.
        From there, you select the right node for the next step.
        Once the that node finishes with the task, you select the next node for the next step
        Once you are done with all the steps, you finalize the summary of the automation process.
        Keep track of all the steps and you have the clear undesrtanding of the each step and where we are in the process.
       
        Example 1: 
        * If the user query is "Open a web browser and naviagate to scholar.google.com and seraach for 'OpenAI'".
        * Then you create plan for the web automation based on the user query.The select the node for the next step. 
        * In this case, you select the omni_parser node for taking the screenshot of the screen to navigate.
        * Now, you will have all the screen component details, you select the gui_action node to click on the browser to open it.
        * Now, you have to select the next step which will be calling omni_parser node to take the screenshot of the screen to understand where we are in the process
        * Then, you call the gui_action node to click on the address bar and type scholar.google.com and press enter.
        * You keep on doing this untill you reach the end of the process.
        """

        messages = state['messages']
        user_query = state['user_query']

        llm = OpenAI(
            #api_key=os.environ.get("GEMINI_API_KEY"),
            #base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        if len(messages) == 0:
            # We only need to give the steps in deatils once. So this section only called first time and this json sceme request 
            # additional deatils called steps_in_details.
            response_json_format = response_json_supervisor
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_query}
            ]
        else:
            # No need to call steps_in details parameter in every request.
            response_json_format = response_json_supervisor_wo_steps_details
            
        completion = llm.chat.completions.create(
            #model='gemini-2.0-pro-exp-02-05',
            model='gpt-4o-mini',
            messages=messages,
            response_format=response_json_format
        )

        return {'messages': [completion.choices[0].message.content], 'sender': ['supervisor']}

    def finalize_summary_node(self, state:WebAgentState):
        pass

    def run(self, user_query=None):
        #label_coordinates, parsed_content_list = self.omni_parser2.parse()
        initial_state = WebAgentState()
        initial_state['user_query'] = user_query
        result = self.graph.invoke(initial_state, config=self.graph_config)
        print(result)

    
        
    