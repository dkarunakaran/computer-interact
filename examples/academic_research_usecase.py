from web_operator.supervisor import Supervisor
from dotenv import load_dotenv

#**************************************READ*************************************************
# This use case shows how to get the title and link of the papers related to the user query. 
# Currently, it looks for the kewords in three databases: arxiv, pubmed, and google scholar.


load_dotenv()  
required_agents = ['research_agent']
supervisor = Supervisor(required_agents=required_agents)
# Make sure you change the config before the configure method
supervisor.config['browser_agent']['headless'] = False
supervisor.config['research_agent']['verbose'] = True
supervisor.config['browser_agent']['verbose'] = True

# Configure the supervisor for automation task
supervisor.configure()
supervisor.run(query=supervisor.prompt_by_usecase(usecase='research', keyword='insurance use case connected vehicles'))

print(supervisor.get_results())






