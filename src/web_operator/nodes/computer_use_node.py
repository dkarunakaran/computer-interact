
from openai import OpenAI
import pyautogui
import base64
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

# Ref: https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb


class ComputerUseNode:
    def __init__(self):
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
        self.schema = """
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "target": {"type": "string"}
                },
                "required": ["action", "target"]
            }
        """
        self.steps_llm = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )

        
    def run(self, user_query=None):
        
        steps = self.get_steps(user_query=user_query)

        print(steps)

        # Gooing through each steps
        for count, dict in enumerate(steps["steps"]):
            phrase_parts = []
            # Iterate through all keys
            for key, value in dict.items():
                phrase_parts.append(str(value).lower())  # Convert values to lowercase strings

            phrase = " ".join(phrase_parts)

            print(f"Step {count}: {phrase}")
        
        #pyautogui.screenshot('my_screenshot.png')
        #screenshot = "my_screenshot.png"
        #output_text, action, display_image = self.perform_gui_grounding(screenshot, user_query)
        #display_image.save('test.png')
        # Display results
        #print(action)


    def get_steps(self, user_query=None):

        completion = self.steps_llm.chat.completions.create(
            model="phi4",
            messages=[
                {"role": "system", "content": """
                You are a computer use assistant and has the capability do the browser automations.
                Create a step by step approach a human would action it to acheive this in GUI. 
                DO NOT have wait, locate, launch, maximize and focus steps. 
                Ouptput in json format. Make sure you have 'steps' key in the json object.
                """
                },
                {"role": "user", "content": [{"type":"text", "text":user_query}]}
            ],
            response_format={ "type": "json_object", "schema" : self.schema }

        )

        steps = json.loads(completion.choices[0].message.content)
        
        return steps


    def perform_gui_grounding(self, screenshot_path, user_query):
        """
        Perform GUI grounding using Qwen model to interpret user query on a screenshot.
        
        Args:
            screenshot_path (str): Path to the screenshot image
            user_query (str): User's query/instruction
            model: Preloaded Qwen model
            processor: Preloaded Qwen processor
            
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
                Message(role="user", content=[
                    ContentItem(text=user_query),
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
        action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        display_image = input_image.resize((resized_width, resized_height))
        display_image = draw_point(input_image, action['arguments']['coordinate'], color='green')
        
        return output_text, action, display_image