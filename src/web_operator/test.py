import subprocess
from openai import OpenAI
from dotenv import load_dotenv
import pyautogui
import base64
import json
import time
import os
from PIL import Image as PILImage


#subprocess.check_output(['ls', '-l'])  # All that is technically needed...
#print(subprocess.check_output(['xdotool', 'mousemove', '--sync', '550', '50']))

#model="gpt-4o-mini"

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


pyautogui.screenshot('my_screenshot.png')
with open('my_screenshot.png', "rb") as f:
    base64_screenshot = base64.b64encode(f.read()).decode("utf-8")


tools = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click on an item on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A description of what you want to click on"
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            },
            "strict": True
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": "type the text on the screen",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A description of what you want to type"
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "press a keyboard key",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A description of whick key you want to press"
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


schema = """
{
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "target": {"type": "string"}
    },
    "required": ["action", "target"]
}
"""


query1 = "Open the firefox webbrowser and naviagate to scholar.google.com"
query2 = "Search for 'OpenAI' in the search bar"

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": """
         You are a computer use assistant and has the capability do the browser automations.
         Create a step by step approach a human would action it to acheive this in GUI(linux). 
         The GUI already have the browser icon. Do not add wait, launch, and focus actions. 
         Ouptput in json format.
         """
    },
        {"role": "user", "content": [
            {"type":"text", "text":query2},
        ]}
    ],
    response_format={ "type": "json_object", "schema" : schema }

)

steps = json.loads(completion.choices[0].message.content)
print(steps)
exit()
for each in steps["steps"]:
    phrase_parts = []
    # Iterate through all keys
    for key, value in each.items():
        phrase_parts.append(str(value).lower())  # Convert values to lowercase strings

    phrase = " ".join(phrase_parts)

    print(phrase)

    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {"role": "user", "content": [
                {"type":"image_url", "image_url":{"url": f"data:image/jpeg;base64,{base64_screenshot}"}},
                {"type":"text", "text":"Here are the contents of the screen."},
                {"type":"text", "text": "What would be the the action for this request:"+ phrase},
            ]}
        ],
        tools = tools
    )

    print(completion.choices[0].message.content)
    print(completion.choices[0].message.tool_calls)
    print("Waiting...")
    time.sleep(10)