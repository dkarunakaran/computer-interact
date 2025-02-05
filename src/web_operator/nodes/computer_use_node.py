
from openai import OpenAI
import pyautogui
import base64
import json
import time
import os
import re
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

        OSATLAS_HUGGINGFACE_SOURCE = "maxiw/OS-ATLAS"
        self.OSATLAS_HUGGINGFACE_MODEL = "OS-Copilot/OS-Atlas-Base-7B"
        self.OSATLAS_HUGGINGFACE_API = "/run_example"

        self.huggingface_client = Client(OSATLAS_HUGGINGFACE_SOURCE)

    def run(self, query):
        pyautogui.screenshot('my_screenshot.png')
        with open('my_screenshot.png', "rb") as f:
           base64_screenshot = base64.b64encode(f.read()).decode("utf-8")

        completion = self.llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": """
                You are a computer use assistant and has the capability do the browser automations.
                Create a step by step approach a human would action it to acheive this in GUI(linux). 
                The GUI already have the browser icon. Do not add wait, launch, and focus actions. 
                If you need to type in a location, we need to locate.
                Ouptput in json format. Make sure you have 'steps' key in the json object.
                """
                },
                {"role": "user", "content": [{"type":"text", "text":query}]}
            ],
            response_format={ "type": "json_object", "schema" : self.schema }

        )
        steps = json.loads(completion.choices[0].message.content)
        print(steps)
        print("Wait for 5 seconds for next action")
        time.sleep(5)

        # Gooing through each steps
        for each in steps["steps"]:
            phrase_parts = []
            # Iterate through all keys
            for key, value in each.items():
                phrase_parts.append(str(value).lower())  # Convert values to lowercase strings

            phrase = " ".join(phrase_parts)

            print(f"Step: {phrase} started")

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
            tool_calls = completion.choices[0].message.tool_calls
            
            if tool_calls:            
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    print("Running this tool: "+function_name)
                    if function_name == "click":
                        query = json.loads(tool_call.function.arguments)["query"]
                        result_x, result_y = self.get_coordinates(query)
                        print(f"query: {query} and position: {result_x}, {result_y}")
                        pyautogui.click(x=result_x, y=result_y) 
                        pyautogui.moveTo(result_x, result_y)

                    if function_name == "type_url":
                        query = json.loads(tool_call.function.arguments)["query"]
                        print(f"query: {query}")
                        pyautogui.hotkey('ctrl', 'a')
                        pyautogui.write(query)    
                        #pyautogui.press('enter')

                    if function_name == "press_key":
                        query = json.loads(tool_call.function.arguments)["query"]
                        print(f"query: {query}")
                        pyautogui.press(query)

                        

            print("Waiting...")
            time.sleep(10)
    

    def get_coordinates(self, query):

        pyautogui.screenshot('my_screenshot.png')

        result = self.huggingface_client.predict(
            image=handle_file("my_screenshot.png"),
            text_input= query + "\nReturn the response in the form of a bbox",
            model_id=self.OSATLAS_HUGGINGFACE_MODEL,
            api_name=self.OSATLAS_HUGGINGFACE_API,
        )
        print(result)

        numbers = [float(number) for number in re.findall(r"\d+\.\d+", result[1])]
        # x1, y1, x2, y2

        result_x, result_y = (numbers[0] + numbers[2]) // 2, (numbers[1] + numbers[3]) // 2
        return result_x, result_y
