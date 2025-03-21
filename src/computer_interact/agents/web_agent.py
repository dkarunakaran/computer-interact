from openai import OpenAI
import os
import time
import torch
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl import smart_resize
from computer_interact.nodes.tools import ComputerUse
from computer_interact.utils import draw_point
import json
import pyautogui
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, List
from langgraph.graph import StateGraph, START, END
from computer_interact.state.web_agent_state import WebAgentState
import  computer_interact.agents.prompts as prompts 


# Ref: https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb

class WebAgent:
    def __init__(self, logger, config = None):
        self.logger = logger
        self.config = config
        model_path = self.config['computer_use_model']
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.graph_config = {"configurable": {"thread_id": "1", "recursion_limit": 20}}
        # Add nodes and edges 
        workflow = StateGraph(WebAgentState)
        workflow.add_node("llm", self.llm_node)
        workflow.add_node("annotate", self.annotate_node)
        workflow.add_node("gui_action", self.gui_action_node)
        workflow.add_node("finalize_summary", self.finalize_summary_node)

        # Add edges
        workflow.add_edge(START, "llm")
        workflow.add_edge("annotate", "gui_action")
        workflow.add_conditional_edges(
            "llm", 
            self.router
        )
        workflow.add_edge(
            "gui_action",
            "llm"
        )
        workflow.add_edge("finalize_summary", END)

        # Set up memory
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        self.screen_width, self.screen_height = pyautogui.size()


    # Define the function that determines whether to continue or not
    def router(self, state: WebAgentState) -> Literal["annotate", "finalize_summary", END]:

        action = json.loads(state['action_to_be_taken'])

        print("---------------")
        print(action['Node'])
        print("---------------")

        # If the LLM makes a tool call, then we route to the "tools" node
        if "annotate" in action['Node']:
            return "annotate"
        if "finalize_summary" in action['Node']:
            return "finalize_summary"
        
        # Otherwise, we stop (reply to the user)
        return END

    def annotate_node(self, state:WebAgentState):

        self.logger.info("annotate node started...")
        pyautogui.screenshot('my_screenshot.png')
        screenshot = "my_screenshot.png"
        action = json.loads(state['action_to_be_taken'])
        actions_to_be_taken = action['Action']
        output_text, selected_function = self.perform_gui_grounding(screenshot, query=actions_to_be_taken)

        return {'selected_function': json.dumps(selected_function), 'sender': ['annotate']}
    

    def gui_action_node(self, state:WebAgentState):
        self.logger.info("gui_action node started...")
        action = json.loads(state['action_to_be_taken'])
        action_taken = action['Action']
        selected_function = json.loads(state['selected_function'])
        print(f"Selected function: {selected_function}")
        print(f"Action taken: {action_taken}")

        # Executing the action
        action = selected_function['arguments']["action"]
        if action in ["left_click", "middle_click", "double_click"]:
            coordinate = selected_function['arguments']["coordinate"]
            pyautogui.click(coordinate[0]-2, coordinate[1]+2)  
        elif action == "right_click":
            pyautogui.click(button='right')
        elif action == "type":
            text = selected_function['arguments']["text"]
            pyautogui.write(text, interval=0.25)  
        elif action == "key":
            keys = selected_function['arguments']["keys"]
            if 'ctrl' in keys:
                pyautogui.keyDown('ctrl')
            for key in keys:
                if key != 'ctrl':
                    pyautogui.press(key)
            if 'ctrl' in keys:
                pyautogui.keyUp('ctrl')

        elif action == "mouse_move":
            coordinate = selected_function['arguments']["coordinate"]
            pyautogui.moveTo(coordinate[0], coordinate[1])  
        elif action == "scroll":
            pixels = selected_function['arguments']["pixels"]
            pyautogui.scroll(pixels)

        return {'actions_taken': [action_taken], 'sender': ['gui_action']}
    
    def llm_node(self, state:WebAgentState):

        self.logger.info("llm_node started...")
        system_msg = prompts.system_msg_llm_node_web_agent
        user_query = state['user_query']
        actions_taken = state['actions_taken']
        llm = OpenAI(
            #api_key=os.environ.get("GEMINI_API_KEY"),
            #base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        messages = [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': f"User query: {user_query}"},
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
        
        return {'action_to_be_taken': action, 'sender': ['llm']}
    
    def finalize_summary_node(self, state:WebAgentState):

        return {'final_answer': "", 'sender': ['finalize_summary']}

    def run(self, user_query=None):
        initial_state = WebAgentState()
        initial_state['user_query'] = user_query
        result = self.graph.invoke(initial_state, config=self.graph_config)
        print(result)

    
    def perform_gui_grounding(self, screenshot_path, query, history=[]):
        """
        Perform GUI grounding using Qwen model to interpret user query on a screenshot.
        
        Args:
            screenshot_path (str): Path to the screenshot image
            query (str): query/instruction
            history: History of the conversation
            
        Returns:
            tuple: (output_text, display_image) - Model's output text and annotated image
        """

        # Open and process image
        input_image = Image.open(screenshot_path)
        resized_height, resized_width = smart_resize(
            input_image.height,
            input_image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,
        )
        
        # Initialize computer use function
        computer_use = ComputerUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )

        # Build messages
        message = NousFnCallPrompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=history),
                Message(role="user", content=[
                    ContentItem(text=query),
                    ContentItem(image=f"file://{screenshot_path}")
                ]),
            ],
            functions=[computer_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]

        # Process input
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[input_image], padding=True, return_tensors="pt").to('cuda')

        # Generate output
        output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Parse action and visualize
        selected_function = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        #display_image = input_image.resize((resized_width, resized_height))
        #display_image = draw_point(input_image, action['arguments']['coordinate'], color='green')
        
        return output_text, selected_function

    

    