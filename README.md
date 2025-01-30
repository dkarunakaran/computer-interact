# Web-operator

A library for automating web tasks, built extensively using LangGraph and other tools.

While it leverages browser capabilities, its current functionality is focused on specific web tasks. It's not designed for arbitrary web automation.

This library excels at tasks solvable by its defined set of agents.

## Installation

1. Setup conda enviornment with python 3.12

2. Web-operator and playwright instllation

    ```
    python -m pip install web-operator

    conda install playwright

    pip install playwright

    playwright install

    playwright install-deps 

    playwright install chrome
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

## Basic Usage 
1. Importing Required Modules
```
from web_operator.supervisor import Supervisor
from dotenv import load_dotenv
```
2. Initializing the Supervisor: 
The Supervisor class manages different web agents. We need to specify agents that will be used.
```
load_dotenv()  # Load environment variables
required_agents = [] # if you don't specify any agents, browser agent only works.
supervisor = Supervisor(required_agents=required_agents)

# Configure the supervisor for automation task
supervisor.configure()
```
3. Web Search Operation: 
This example shows how to perform a search on DuckDuckGo:
```
prompt = """
    Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and return the summary of results you get. No need to perform any further actions.
"""
supervisor.run(query=prompt)

# Output
print(supervisor.get_results())
```

## Basic Usage with required agents list
1. Make sure .env file has the location of the json file for authentication Google APIs (use this document that explain how can you get these files in the first place)
2. Initialize the supervisor and other required librarires
```
from web_operator.supervisor import Supervisor
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
required_agents = ['gmail_agent']  # Specify token required agents
supervisor = Supervisor(required_agents=required_agents)

# Configure the supervisor for automation task
supervisor.configure()
```
3. Gmail Email Processing
```
prompt = """
    go to gmail and find email with subject 'Open-Source Rival to OpenAI's Reasoning Model'
    We need only the content of the latest email of the above subject and disgard other emails.
    Extract the first URL (link) from the email content.
    Naviagte to the URL and summarise the content and no further navigation is required

    **Constraints:**
    - Only extract the first URL found in the email body.
    - If no URL is found, return "No URL found."
"""

supervisor.run(query=prompt)

# Output
print(supervisor.get_results())
```

## Basic usage with headless mode turned off
1. Actavte you conda enviorment
2. Run the below command to find the appropriate xvfb library
```
conda search -c conda-forge xvfb
```
3. Select the appropriate xvfb library from the list.
```
# Sample output of step 1
pytest-xvfb                    1.1.0          py27_0  conda-forge                                                            pytest-xvfb                    1.1.0       py36_1000  conda-forge                                                            conda-forge         
xorg-x11-server-xvfb-conda-aarch64          1.20.4   ha134caf_1109  conda-forge                
xorg-x11-server-xvfb-cos6-x86_64          1.17.4               4  pkgs/main           
xorg-x11-server-xvfb-cos6-x86_64          1.17.4      h5c27f9d_0  pkgs/main             
xorg-x11-server-xvfb-cos7-ppc64le          1.20.4    ha826a6f_103  conda-forge         
xvfbwrapper                    0.2.9 py36h9f0ad1d_1002  conda-forge         
xvfbwrapper                    0.2.9 py36h9f0ad1d_1003  conda-forge         
xvfbwrapper                    0.2.9 py36hc560c46_1003  conda-forge         
xvfbwrapper                    0.2.9       py37_1000  conda-forge         
xvfbwrapper                    0.2.9       py37_1001  conda-forge         
xvfbwrapper                    0.2.9 py37h89c1867_1004  conda-forge         
xvfbwrapper                    0.2.9 py37hc8dfbb8_1002  conda-forge         
xvfbwrapper                    0.2.9 py37hc8dfbb8_1003  conda-forge          
```
4. For example, we have ubuntu intel system and installed the below library
```
conda install -c conda-forge xorg-x11-server-xvfb-conda-x86_64
```
5. Run xvfb-run your-command
```
xvfb-run python test.py
```


## How to change the basic config

1. print config 
```
print(supervisor.config)

#Typical output
{'debug': False, 'GOOGLE_API': {'scopes': ['https://mail.google.com/', 'https://www.googleapis.com/auth/tasks', 'https://www.googleapis.com/auth/drive']}, 'gmail_agent': {'recursion_limit': 10, 'verbose': False}, 'browser_agent': {'recursion_limit': 10, 'verbose': False}, 'supervisor': {'recursion_limit': 10}}

```
2. modify
```
supervisor.config["debug"] = True
print(supervisor.config)

#Typical output
{'debug': True, 'GOOGLE_API': {'scopes': ['https://mail.google.com/', 'https://www.googleapis.com/auth/tasks', 'https://www.googleapis.com/auth/drive']}, 'gmail_agent': {'recursion_limit': 10, 'verbose': False}, 'browser_agent': {'recursion_limit': 10, 'verbose': False}, 'supervisor': {'recursion_limit': 10}}

```
3. Sample full code
```
load_dotenv()  
required_agents = []
supervisor = Supervisor(required_agents=required_agents)

# Make sure you change the config before the configure method
supervisor.config['GMAIL_AGENT']['verbose'] = True # verbose for displaying detailed logs of agents' tasks
supervisor.config['BROWSER_AGENT']['verbose'] = True

# Configure the supervisor for automation task
supervisor.configure()
supervisor.run(query=prompt2)

# Output
print(supervisor.get_results())
```

## Best OPENAI model
We tested with different GPT models, and GPT-4 is the best among them. By default, GPT-4o-mini is used. To change the model, follow the code below:
```
supervisor.config['model'] = 'gpt-4o'
```

## Content of .env file
```
OPENAI_API_KEY=your_token
GOOGLE_API_CREDS_LOC=path to your credentials.json
GOOGLE_API_TOKEN_LOC=path to token.json
```

## Available Agents

1. **browse_agent**:
* Provides web search capabilities.
* Active by default.
2. **gmail_agent**:
* Enables Gmail operations.
* Not active by default.
* Requires a Google API token.
* Must be specified in required_agents when initializing the supervisor.
3. **arxiv_agent**:
* Searches the Arxiv paper based on the user input.
* Not active by default.
* No API token is required.
* Must be specified in required_agents when initializing the supervisor.
