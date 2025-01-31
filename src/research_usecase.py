from web_operator.supervisor import Supervisor
from dotenv import load_dotenv


prompt1 = """
        get all papers with title, abstract, and url from arxiv that touches the topic called insurance usecase of connected vehicles. 
        Make sure you tried different keywords related to the topic. 
        Finally, Go to https://scholar.google.com.au and get the result for insurance usecases of connected vehicles.
        Then give me a comibed result. 
        once you done it, return FINISH
"""

prompt2 = """
        Go to https://scholar.google.com.au and search for 'insurance usecases of connected vehicles'.
        Get the results with title and url.
        Naviagate to second page as well.
        once you done it, return FINISH
"""

prompt3 = """
        get all papers with title, abstract, and url from arxiv and pubmed that touches the topic called insurance usecase of connected vehicles. 
        Make sure you tried different keywords related to the topic. 
        Go to https://scholar.google.com.au and search for 'insurance usecases of connected vehicles'.
        Get the results with title and url.
        Naviagate to second page as well.
        Then give me a comibed result. 
        once you done it, return FINISH
"""

load_dotenv()  
required_agents = ['research_agent']
supervisor = Supervisor(required_agents=required_agents)
# Make sure you change the config before the configure method
supervisor.config['browser_agent']['headless'] = False
supervisor.config['research_agent']['verbose'] = True
supervisor.config['browser_agent']['verbose'] = True

# Configure the supervisor for automation task
supervisor.configure()
supervisor.run(query=prompt2)
print(supervisor.get_results())






