# Web-operator

A library for automating web tasks, built extensively using LangGraph and other tools.

While it leverages browser capabilities, its current functionality is focused on specific web tasks. It's not designed for arbitrary web automation.

This library excels at tasks solvable by its defined set of agents.

## Installation

1. Setup conda enviornment with python 3.12

2. Web-operator and other software installation

    ```
    python -m pip install web-operator
    python -m pip install --no-cache-dir torch==2.4.0 torchvision==0.19 torchaudio==2.4.0 xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install git+https://github.com/huggingface/transformers
    
    ```

3. Environment Setup

    This guide explains how to set up and manage environment variables for your project using python-dotenv.

    a. Install the python-dotenv library using pip:

    ```
    pip install python-dotenv
    ```
    b. Create a .env file in your project's root directory with the following structure:
    ```
    OPENAI_API_KEY=your_openai_api_key

    # Only add below config if you want to use the GOOGLE services
    GOOGLE_API_CREDS_LOC=your credentials.json file location
    GOOGLE_API_TOKEN_LOC=your token.json file location
    ```

    c. Add .env to your .gitignore file to prevent accidentally committing sensitive information:

    d. code for load environment variables 
    ```
    from dotenv import load_dotenv
    import os
    load_dotenv()
    ```


