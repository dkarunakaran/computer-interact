import subprocess
from openai import OpenAI
from dotenv import load_dotenv
import pyautogui
import base64
import json
import time
import os
from PIL import Image as PILImage
from gradio_client import Client, handle_file
import re
from web_operator.nodes.computer_use_node import ComputerUseNode
from web_operator.nodes import router_node


load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
computerNode = ComputerUseNode()

query1 = "Open a web browser and naviagate to scholar.google.com"
query2 = "Search for 'OpenAI' in the search bar"
query3 = "go to gmail using google API and read the email titles 'test'"
query4 = "Nothing"

query = query1
# Select the right node
# Adding more than 12 tools to a llm will degrade the performance. So, I want to seperate 
# API based services vs pure browser lookup automation.

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": """
         You are a supervisor and tasked to select the right node for further automation. 
         You have two nodes: computer_use_node and api_operation_node.
         Given, the user prompt select the right node. 
         If there is no right match, then don't return any node.
         """
        },
        {
            "role": "user", "content": [
                {"type":"text", "text":query},
            ]
        }
    ],
    tools = router_node.tools
)

print(completion.choices[0].message.tool_calls)
node_selected = completion.choices[0].message.tool_calls

if node_selected:
    node_name = node_selected[0].function.name
    if node_name == 'computer_use_node':
        computerNode.run(query=query)

else:
    print("No tools are selected")
    
    
exit()





OSATLAS_HUGGINGFACE_SOURCE = "maxiw/OS-ATLAS"
OSATLAS_HUGGINGFACE_MODEL = "OS-Copilot/OS-Atlas-Base-7B"
OSATLAS_HUGGINGFACE_API = "/run_example"

huggingface_client = Client(OSATLAS_HUGGINGFACE_SOURCE)

pyautogui.screenshot('my_screenshot.png')

result = huggingface_client.predict(
    image=handle_file("my_screenshot.png"),
    text_input= "Web browser" + "\nReturn the response in the form of a bbox",
    model_id=OSATLAS_HUGGINGFACE_MODEL,
    api_name=OSATLAS_HUGGINGFACE_API,
)
print(result)