from openai import OpenAI
from web_operator.nodes.computer_use_node import ComputerUseNode
from web_operator.nodes.api_operation_node import APIOperationNode
from web_operator.nodes import router_node
import os
from web_operator.utils import logger_helper


class Supervisor:
    def __init__(self):
        self.config = self.__get_config()
        # __ adding infront of the variable and method make them private
        self.__logger = logger_helper(self.config)
        if not os.environ.get("OPENAI_API_KEY"):
            raise KeyError("OPENAI API token is missing, please provide it .env file.") 
        
        self.llm = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        )

    def configure(self):
        self.computerUseNode = ComputerUseNode()
        self.apiOperationNode = APIOperationNode()

    def __get_config(self):
        """
        This function defines the config for the library
        """
        cfg = {
            'debug': False,
            'model': 'gpt-4o-mini'
        }

        return cfg

    
    def run(self, user_query=None):
        completion = self.llm.chat.completions.create(
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
                    "role": "user", "content": [{"type":"text", "text":user_query}]
                }
            ],
            tools = router_node.tools
        )

        node_selected = completion.choices[0].message.tool_calls
        self.__logger.info(node_selected)
        if node_selected:
            for node in node_selected:
                node_name = node.function.name
                if node_name == 'api_operation_node':
                    result = self.apiOperationNode.run(user_query=user_query)
                    self.__logger.debug(result)
                if node_name == 'computer_use_node':
                    self.computerUseNode.run(user_query=user_query)
        else:
            self.__logger.info("No nodes are selected")

        

if __name__ == "__main__":
    supervisor = Supervisor()
    
    

   