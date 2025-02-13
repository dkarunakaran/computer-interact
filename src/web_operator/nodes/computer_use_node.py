
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
from web_operator.nodes.tools import ComputerUse
from web_operator.utils import draw_point
import json
import pyautogui

# Ref: https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb

class ComputerUseNode:
    def __init__(self, logger, config = None):
        self.logger = logger
        self.config = config
        model_path = self.config['computer_use_model']
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
        self.response_json_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "step_reasoning",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string"},
                                    #"target": {"type": "string"},
                                    "description": {"type": "string"},
                                    "device": {"type": "string"}
                                },
                                "required": ["action", "description", "device"],
                                "additionalProperties": False
                            }
                        },
                    },
                    "required": ["steps"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    
        self.steps_llm = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        self.message = {
            'input':[],
            'action': [],
            'url': []
        }

        
    def run(self, user_query=None):
        history = []
        steps = self.get_steps(user_query=user_query)
        self.logger.info(steps)
        # Going through each steps
        for count, dict in enumerate(steps["steps"]):
            phrase = dict['description'] + " using " + dict['device']
            time.sleep(1)
            self.logger.info(f"Step {count}: {phrase}")
            pyautogui.screenshot('my_screenshot.png')
            screenshot = "my_screenshot.png"
            output_text, selected_function = self.perform_gui_grounding(screenshot, query=phrase, history=history)
            # Display results
            self.logger.info(selected_function)
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

            history.append(ContentItem(text=phrase+str(selected_function)))
            

    def get_steps(self, user_query=None):

        completion = self.steps_llm.chat.completions.create(
            model=self.config['step_creation_model'],
            messages=[
                {"role": "system", "content": """
                You are a computer use assistant and has the capability do the browser automations.
                Create a step by step approach a human would action it to acheive this in GUI. 
                DO NOT have wait, locate, launch, maximize and focus steps. 
                DO NOT combine steps together.
                Make sure you have 'steps' key in the json object.
                
                Example format for the userquery 'Open a web browser and naviagate to scholar.google.com':
                {'steps': [
                 {'action': 'click', 'description': 'Click the Chrome web browser', 'device': 'mouse'}, 
                 {'action': 'type', 'description': 'In the address bar, type 'scholar.google.com' ', 'device': 'keyboard'}, 
                 {'action': 'press enter', 'description': 'Press the Enter key to navigate to Google Scholar', 'device': 'keyboard'}
                 ]
                }
                """
                },
                {"role": "user", "content": [{"type":"text", "text":user_query}]}
            ],
            response_format=self.response_json_format

        )

        steps = json.loads(completion.choices[0].message.content)
        
        return steps


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