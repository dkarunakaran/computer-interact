from web_operator.supervisor import Supervisor
from dotenv import load_dotenv

prompt = """
            Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and return the summary of results you get. Use fill tool to fill in fields and print out url at each step.
        """
load_dotenv()  
token_required_agents = []
supervisor = Supervisor(token_required_agents=token_required_agents)
supervisor.run(query=prompt)