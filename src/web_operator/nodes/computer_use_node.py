
from openai import OpenAI
import pyautogui
import base64
import json
import time
import os
from gradio_client import Client, handle_file
from web_operator.nodes.tools import computeruse_tools


class ComputerUseNode:
    def __init__(self):
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

        self.llm = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

    def run(self, query):
        #pyautogui.screenshot('my_screenshot.png')
        with open('my_screenshot.png', "rb") as f:
           base64_screenshot = base64.b64encode(f.read()).decode("utf-8")

        completion = self.llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": """
                You are a computer use assistant and has the capability do the browser automations.
                Create a step by step approach a human would action it to acheive this in GUI(linux). 
                The GUI already have the browser icon. Do not add wait, launch, and focus actions. 
                Ouptput in json format.
                """
                },
                {"role": "user", "content": [{"type":"text", "text":query}]}
            ],
            response_format={ "type": "json_object", "schema" : self.schema }

        )
        steps = json.loads(completion.choices[0].message.content)
        print(steps)
        # Gooing through each steps
        for each in steps["steps"]:
            phrase_parts = []
            # Iterate through all keys
            for key, value in each.items():
                phrase_parts.append(str(value).lower())  # Convert values to lowercase strings

            phrase = " ".join(phrase_parts)

            print(phrase)

            completion = self.llm.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {"role": "user", "content": [
                        {"type":"image_url", "image_url":{"url": f"data:image/jpeg;base64,{base64_screenshot}"}},
                        {"type":"text", "text":"Here are the contents of the screen."},
                        {"type":"text", "text": "What would be the the action for this request:"+ phrase},
                    ]}
                ],
                tools = computeruse_tools
            )
            print(completion.choices[0].message.tool_calls)
            break
            print("Waiting...")
            time.sleep(10)
