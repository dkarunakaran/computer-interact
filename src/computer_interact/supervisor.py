from openai import OpenAI
from computer_interact.agents.web_agent import WebAgent
import os
from computer_interact.utils import logger_helper
from computer_interact.config_file import Config
import computer_interact.agents.prompts as prompts 
from computer_interact.tools import superviser_router


class Supervisor:
    def __init__(self):
        self.config = Config()
        # __ adding infront of the variable and method make them private
        self.logger = logger_helper(self.config)
        if not os.environ.get(self.config.llm_api_key_name):
            raise KeyError(f"{self.config.llm_api_key_name} is missing, please provide it .env file.")
        self.logger.info(f"------------LLM selected: {self.config.llm}------------")
        self.llm = OpenAI(
            api_key=os.environ.get(self.config.llm_api_key_name),
            base_url=self.config.llm_base_url
        )
        self.state = []

    def configure(self):
        self.web_agent = WebAgent(logger=self.logger, config=self.config)
    
    def run(self, user_query=None):
        messages = [
            {'role': 'system', 'content': prompts.system_msg_agent_selector},
            {'role': 'user', 'content': [{"type":"text", "text":user_query}]}
        ]
        completion = self.llm.chat.completions.create(
            model=self.config.llm,
            messages=messages,
            tools=superviser_router.tools
        )
        node_selected = completion.choices[0].message.tool_calls
        self.logger.info(node_selected)
        if node_selected:
            for node in node_selected:
                node_name = node.function.name
                if node_name == 'os_agent':
                    self.logger.debug("os_agent selected")
                if node_name == 'web_agent':
                    self.logger.debug("web_agent selected")
                    self.web_agent.run(user_query=user_query)
        else:
            self.logger.info("No nodes are selected")

if __name__ == "__main__":
    supervisor = Supervisor()
    
    

   