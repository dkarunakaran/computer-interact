# Web-operator

A library for automating web tasks, built extensively using LangGraph and other tools.

While it leverages browser capabilities, its current functionality is focused on specific web tasks. It's not designed for arbitrary web automation.

This library excels at tasks solvable by its defined set of agents.

# Requirement
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

    c. Add .env to your .gitignore file to prevent accidentally committing sensitive information:

    d. code for load environment variables 
    ```
    from dotenv import load_dotenv
    import os
    load_dotenv()
    ```


