from web_operator.supervisor import Supervisor
from web_operator.utils import google_api_authenticate
from dotenv import load_dotenv

prompt1 = """
        go to gmail and find first email with subject 'Timesheets to sign for Kids Early Learning Family Day Care - Blacktown City'
        We need only the content of the latest email of the above subject and disgard other emails.
        Extract the first URL (link) from the email content.
        Naviagte to the URL and then login using password. 
        Then, summarise the content of the page
        Once you done that return FINISH
        """


prompt2 = """
            Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and get the result and summarise it. Once you done that return FINISH.
        """

prompt3 ="""

        Go to booking.com, find the cheapest hotel around sydney CBD. 
        Close the signin option, if they show up. 
        Just look at the first page only.
        Once you done that return FINISH
"""

prompt4 ="""
        get me 10 papers that touches the insurance usecase of connected vehicles. then summarise all of them together. Once you done that return FINISH"
"""

load_dotenv()  
required_agents = ['gmail_agent', 'arxiv_agent']
supervisor = Supervisor(required_agents=required_agents)
# Make sure you change the config before the configure method
supervisor.config['browser_agent']['headless'] = False
supervisor.config['arxiv_agent']['verbose'] = True
supervisor.config['browser_agent']['verbose'] = True

# Configure the supervisor for automation task
supervisor.configure()
supervisor.run(query=prompt4)
print(supervisor.get_results())



