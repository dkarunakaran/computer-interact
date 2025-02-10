from openai import OpenAI
from web_operator.nodes.computer_use_node import ComputerUseNode
from web_operator.nodes.api_operation_node import APIOperationNode
from web_operator.nodes import router_node
import os
from dotenv import load_dotenv


load_dotenv()  

if not os.environ.get("OPENAI_API_KEY"):
    raise KeyError("OPENAI API token is missing, please provide it .env file.") 


computerUseNode = ComputerUseNode()
apiOperationNode = APIOperationNode()

query1 = "Open a web browser and naviagate to scholar.google.com"
query2 = "Search for 'OpenAI' in the search bar"
query3 = "go to gmail using google API and read the email titles 'test'"
query4 = "Open the settings in the os and set volume to zero"
query5 = "Open a firefox web browser and  type sportsbet.com using and press enter"
query6 = "Open a firefox web browser and  type scholar.google.com and enter and then search for 'OpenAI'"

""""
Prompt Design:

The best prompt will be if you specify each actino step by step. 
For example, if you need to open a web browser and navigate to scholar.google.com and search for openai, 
you can specify each step as follows:

action 1: open the  firfox web browser
action 2: type scholar.google.com 
action 3: press enter for search
action 4: type openai in the search box of google scholar
action 5: press enter for search
"""

query7 = """
action 0: Using gmail API, go to gmail and read the email titles 'test'
action 1: open the firfox web browser
action 2: type scholar.google.com 
action 3: press enter for search
action 4: type openai in the search box of google scholar
action 5: press enter for search
"""

user_query = query7

"""
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)
"""

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

completion = client.chat.completions.create(
    #model="llama3.1:latest",
    model="gpt-4o-mini",
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
            {"type":"text", "text":user_query},
            ]
        }
    ],
    tools = router_node.tools
)

node_selected = completion.choices[0].message.tool_calls
print(node_selected)
if node_selected:
    for node in node_selected:
        node_name = node.function.name
        if node_name == 'api_operation_node':
            result = apiOperationNode.run(user_query=user_query)
            print(result)
        if node_name == 'computer_use_node':
            computerUseNode.run(user_query=user_query)
else:
    print("No tools are selected")


