import subprocess
from openai import OpenAI
from dotenv import load_dotenv
import pyautogui
import base64
import json
import time
import os
from PIL import Image
import requests
from gradio_client import Client, handle_file
import re
from web_operator.nodes.computer_use_node import ComputerUseNode
from web_operator.nodes import router_node
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

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
    #pyautogui.screenshot('my_screenshot.png')
    with open('my_screenshot.png', "rb") as f:
        base64_screenshot = base64.b64encode(f.read()).decode("utf-8")

else:
    print("No tools are selected")
    
    
exit()
