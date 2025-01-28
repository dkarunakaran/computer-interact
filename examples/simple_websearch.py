from web_operator.supervisor import Supervisor
from dotenv import load_dotenv

prompt = """
    Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and return the summary of results you get. No need to perform any further actions.
"""
load_dotenv()  
token_required_agents = []
supervisor = Supervisor(token_required_agents=token_required_agents)

# Make sure you change the config before the configure method
supervisor.config['BROWSER_AGENT']['verbose'] = True

# Configure the supervisor for automation task
supervisor.configure()
supervisor.run(query=prompt)

print(supervisor.get_results())