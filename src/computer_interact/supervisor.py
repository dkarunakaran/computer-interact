from openai import OpenAI
from computer_interact.agents.web_agent import WebAgent
import os
from computer_interact.utils import logger_helper


class Supervisor:
    def __init__(self):
        self.config = self.__get_config()
        # __ adding infront of the variable and method make them private
        self.logger = logger_helper(self.config)
        """if not os.environ.get("OPENAI_API_KEY"):
            raise KeyError("OPENAI API token is missing, please provide it .env file.")"""
        if not os.environ.get("GEMINI_API_KEY"):
            raise KeyError("GEMINI API token is missing, please provide it .env file.") 
        
        self.llm = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        self.state = []

    def configure(self):
        self.web_agent = WebAgent(logger=self.logger, config=self.config)

    def __get_config(self):
        """
        This function defines the config for the library
        """
        cfg = {
            'debug': False,
            'step_creation_model': 'gemini-2.0-pro-exp-02-05',
            'computer_use_model': 'Qwen/Qwen2.5-VL-7B-Instruct'
        }

        return cfg

    
    def run(self, user_query=None):
        self.web_agent.run(user_query=user_query)

        

if __name__ == "__main__":
    supervisor = Supervisor()
    
    

   