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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
# Quantization - fp4 (4-bit Floating-Point)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_quant_type="nf4" 
)

# Quantization - q4 (4-bit Integer)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_quant_type="int4" 
)


# Quantization - q8 (8-bit Integer)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=1.0,  # Adjust this threshold as needed
    llm_int8_enable_fp32_grad=True,
    llm_int8_keep_original_type=True
)
'''

processor = AutoProcessor.from_pretrained("bytedance-research/UI-TARS-72B-DPO")
model = AutoModelForImageTextToText.from_pretrained("bytedance-research/UI-TARS-72B-DPO")
tokenizer = AutoTokenizer.from_pretrained("bytedance-research/UI-TARS-72B-DPO")


# Image
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open('my_screenshot.png')

conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
            },
            {
                "type":"text",
                "text":"Open a web browser and naviagate to scholar.google.com"
            }
        ]
    }
]

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
#inputs = inputs.to(device)
#model = model.to(device)

# Inference: Generation of the output
generate_ids = model.generate(**inputs, max_length=2800)
output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)
