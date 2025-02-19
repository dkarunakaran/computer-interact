from openai import OpenAI
from computer_interact.tools import superviser_router_tools
from computer_interact.utils import logger_helper
from computer_interact.config_file import Config
from computer_interact.agents.web_agent import WebAgent
from computer_interact.agents.os_agent import OSAgent
import os
import asyncio

# Ref 1: https://huggingface.co/microsoft/OmniParser-v2.0
# Ref 2: https://github.com/microsoft/OmniParser/tree/master
# Ref 3: https://github.com/microsoft/OmniParser/blob/master/demo.ipynb 

class Supervisor:
    def __init__(self):
        self.config = Config()
        # __ adding infront of the variable and method make them private
        self.logger = logger_helper(self.config)
        if not os.environ.get("GEMINI_API_KEY"):
            raise KeyError("GEMINI API token is missing, please provide it .env file.") 
        
        self.llm = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        self.state = []

    def configure(self):
        self.web_agent = WebAgent(logger=self.logger, config=self.config)
        self.os_agent = OSAgent(logger=self.logger, config=self.config)

    
    def run(self, user_query=None):
        asyncio.run(self.web_agent.run(user_query=user_query))
        '''completion = self.llm.chat.completions.create(
            model=self.config.agent_selector_model,
            messages=[
                {"role": "system", "content": """
                You are a supervisor and tasked to select the right node for further automation. 
                You have two nodes: computer_use_node and api_operation_node.
                Given, the user prompt select the right node. 
                If there is no right match, then don't return any node.
                """
                },
                {
                    "role": "user", "content": [{"type":"text", "text":user_query}]
                }
            ],
            tools = superviser_router_tools.tools
        )

        node_selected = completion.choices[0].message.tool_calls
        if node_selected:
            for node in node_selected:
                node_name = node.function.name
                if node_name == 'os_agent':
                    self.logger.info("OS agent is calling now...")
                if node_name == 'web_agent':
                    asyncio.run(self.web_agent.run(user_query=user_query))
                    
            
        else:
            self.logger.info("No nodes are selected")'''

    

if __name__ == "__main__":
    supervisor = Supervisor()
    
    

   