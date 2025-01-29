from web_operator.supervisor import Supervisor
from dotenv import load_dotenv


prompt2 = """
            Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and get the result and summarise it. Once you get the result, stop the task.
        """

prompt3 ="""

        Go to booking.com, if there is a signin page, close it always and continue. find the cheapest stay around sydney CBD, starting from 29th Jan to 30th Jan. 
"""

load_dotenv()  
token_required_agents = []
supervisor = Supervisor(token_required_agents=token_required_agents)
# Make sure you change the config before the configure method
supervisor.config['BROWSER_AGENT']['headless'] = False
supervisor.config['BROWSER_AGENT']['verbose'] = True

# Configure the supervisor for automation task
supervisor.configure()
supervisor.run(query=prompt3)
#print(supervisor.get_results())



