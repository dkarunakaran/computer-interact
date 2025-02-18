from computer_interact.omni_parser2 import OmniParser2

class WebAgent:

    def __init__(self, logger, config):
        self.config = config
        self.logger = logger
        self.omni_parser2 = OmniParser2(logger=self.logger, config=self.config)
        

    def run(self, user_query=None):
        self.omni_parser2.parse()