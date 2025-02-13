# Web-operator

A library for automating web tasks, built extensively using LangGraph and other tools.

While it leverages browser capabilities, its current functionality is focused on specific web tasks. It's not designed for arbitrary web automation.

This library excels at tasks solvable by its defined set of agents.

## Requirement
This required an Nvidia GPU with 12GB of VRAM to run the Hugging Face model locally. It also required the Gemini API.

## Installation

1. Setup conda enviornment with python 3.12

2. Web-operator and other software installation

    ```
    conda install pytorch pytorch-cuda=12.1 flash-attn=2.6.1
    python -m pip install --no-cache-dir git+https://github.com/huggingface/transformers #
    python -m pip install web-operator
    
    
    ```

3. Environment Setup

    This guide explains how to set up and manage environment variables for your project using python-dotenv.

    a. Install the python-dotenv library using pip:

    ```
    pip install python-dotenv
    ```
    b. Create a .env file in your project's root directory with the following structure:
    ```
    GEMINI_API_KEY=your_openai_api_key
    ```

    c. Add .env to your .gitignore file to prevent accidentally committing sensitive information:

    d. code for load environment variables 
    ```
    from dotenv import load_dotenv
    import os
    load_dotenv()
    ```

## Usecase
please see the examples folder for a simple websearch usecase.

## How to change the basic config

1. print config
```
print(supervisor.config)

#Typical output
{'debug': False, 'step_creation_model': 'gemini-2.0-pro-exp-02-05', 'computer_use_model': 'Qwen/Qwen2.5-VL-7B-Instruct'}
```

2. modify

```
supervisor.config["debug"] = True
print(supervisor.config)

#Typical output
{'debug': False, 'step_creation_model': 'gemini-2.0-pro-exp-02-05', 'computer_use_model': 'Qwen/Qwen2.5-VL-7B-Instruct'}
```

3. Sample full code

```
from web_operator.supervisor import Supervisor
from dotenv import load_dotenv

load_dotenv()  

query = """
action 1: open the firfox web browser
action 1.1: click on the address bar
action 2: type scholar.google.com 
action 3: press enter for search
action 4: type openai in the search box of google scholar
action 5: press enter for search
action 6: click on the address bar
action 7: press ctrl and a
action 7.1:press ctrl and c
action 8: scroll down 
action 9: click on the next button 
action 10: click on the address bar
action 11: press ctrl and a
action 12: press ctrl and c
action 13: close the browser
"""
user_query = query

supervisor = Supervisor()

supervisor.config["debug"] = True

# Make sure the config is changes before the configure function call.
supervisor.configure()
supervisor.run(user_query)


```
