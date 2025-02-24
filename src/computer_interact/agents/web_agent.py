from computer_interact.omni_parser2 import OmniParser2
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
        workflow.add_node("annotate", self.annotate_node)
        """workflow.add_node("gui_action", self.gui_action_node)
        workflow.add_node("finalize_summary", self.finalize_summary_node)"""

        # Add edges
        workflow.add_edge(START, "supervisor_llm")
        workflow.add_conditional_edges(
            "supervisor_llm", 
            self.router
        )
        workflow.add_edge(
            "annotate",
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
    def router(self, state: WebAgentState) -> Literal["annotate", END]:

        messages = state['messages']
        last_message = messages[-1][0]['content']
        steps = json.loads(last_message)
        # If the LLM makes a tool call, then we route to the "tools" node
        if "annotate" in steps['next_step']:
            return "annotate"
        # Otherwise, we stop (reply to the user)
        return END

    def annotate_node(self, state:WebAgentState):

        self.logger.info("omni_parser node started...")
        label_coordinates, parsed_content_list = self.omni_parser2.parse()
        message = [
                {"role": "user", "content": f"Here is the details of screen items and their locations: {str(parsed_content_list)}. Now supervisor needs to make decision what to do from here"}
            ]
        
        return {'messages': message, 'sender': ['annotate']}
        

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
        You are WebRover, an autonomous AI agent designed to browse the web, interact with pages, and extract or aggregate information based on user queries—much like a human browsing the internet. You have access to the following tools:
            - Click Elements: Click on a specified element using its XPath. For links, open them in a new tab.
            - Type in Inputs: Type text into an input field identified by its XPath.
            - Scroll and Read (Scrape+RAG): Scroll down the page while scraping visible text and images to store in a vector database.
            - Close Page: Close the current tab and switch focus to the last opened tab.
            - Wait: Pause for a specified amount of time.
            - Go Back: Navigate back to the previous page.
            - Go to Search: Navigate to Google.

        Your inputs include:
            - The user's query (what the user wants to achieve).
            - A list of icon/text box description on the current page with properties: [type, bbox, interactivity, content, source].
            - A record of actions taken so far.
            - A list of URLs already visited (do not revisit the same URL).

        Your task:
            - Decide the next best action (or a coherent sequence of actions) to move closer to fulfilling the user's request.
            - Evaluate the context by carefully reviewing the user's query, previous actions taken, visited URLs, and conversation history.
            - **Important:** When selecting a DOM element for an action, examine its "text" and "description" fields. For example, if the task is to input a departure date on Google Flights, only choose an input field if its description or visible text includes keywords like "departure", "depart", or "depart date". Do not select a generic input element that lacks specific contextual clues.
            - Avoid repeating the same search or action if it has already been performed without progress. If a search term or action was already attempted and yielded no new information, refine or change your approach.
            - Plan your steps: If multiple sequential actions are needed (e.g., scroll then click, or type a refined search query), output them in order until a page navigation or significant state change occurs.
            - Be natural and precise: Mimic human browsing behavior—click on visible links, type into search bars, scroll to reveal more content, and update your strategy if repeated results occur.

        Action guidelines:
            - Click Elements: Use for selecting links, buttons, or interactive items. If a link has already been followed (or if its URL is in the visited list), avoid re-clicking it.
            - Type in Inputs: Use for entering or refining search queries and form inputs. If the same query has been issued before, consider modifying or extending it.
            - Scroll and Read (Scrape+RAG): Use to gather content when no immediately actionable link is visible.
            - Close Page: Use when you need to exit a tab and return to a previous page.
            - Wait: Use to allow the page sufficient time to load or update after an action.
            - Go Back: Use when you need to return to a previous state or page.
            - Go to Search & WebPage Search: Use these to initiate or refine searches if no better actions are available.
            - Retry: Use only when you are unable to infer the next action from the current context.

        Output format:
            - Clearly output your action(s) in a structured format including:
                - Thought: Your reasoning behind the chosen action(s), considering previous attempts.
                - Action: The action to be taken.
                - Reasoning: Detailed reasoning behind your action.
            - Do not output a repeated search term if it was already used and did not lead to progress; instead, suggest a refined or alternative approach.
            - Only output one coherent action or logical sequence of actions at a time, ensuring each step builds on previous actions logically and naturally.
        """

        messages = state['messages']
        #print(type(messages))
        user_query = state['user_query']

        llm = OpenAI(
            #api_key=os.environ.get("GEMINI_API_KEY"),
            #base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        if len(messages) == 0:
            # We only need to give the steps in deatils once. So this section only called first time and this json sceme request 
            # additional deatils called steps_in_details.
            response_json_format = response_json_supervisor
            message = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_query}
            ]
        else:
            # No need to call steps_in details parameter in every request.
            response_json_format = response_json_supervisor_wo_steps_details
            
            message = [messages[-1]] 
            print(message)
            
        completion = llm.chat.completions.create(
            #model='gemini-2.0-pro-exp-02-05',
            model='gpt-4o-mini',
            messages=message,
            response_format=response_json_format
        )

        print(completion.choices[0].message.content)

        message = [
            {"role": "user", "content": completion.choices[0].message.content}
        ]

        return {'messages': [message], 'sender': ['supervisor']}

    def finalize_summary_node(self, state:WebAgentState):
        pass

    def run(self, user_query=None):
        #label_coordinates, parsed_content_list = self.omni_parser2.parse()
        initial_state = WebAgentState()
        initial_state['user_query'] = user_query
        result = self.graph.invoke(initial_state, config=self.graph_config)
        print(result)

    
        
    