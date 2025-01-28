from web_operator.supervisor import Supervisor
from web_operator.utils import google_api_authenticate
from dotenv import load_dotenv

prompt1 = """
        go to gmail and find email with subject 'Open-Source Rival to OpenAI's Reasoning Model'
        We need only the content of the latest email of the above subject and disgard other emails.
        Extract the first URL (link) from the email content.
        Naviagte to the URL and summarise the content and no further navigation is required

        **Constraints:**
        - Only extract the first URL found in the email body.
        - If no URL is found, return "No URL found."

        """

prompt2 = """
            Go to https://duckduckgo.com, search for insurance usecases in connected vehicles using input box you find from that page, click search button and get the result and summarise it. Once you get the result, stop the task.
        """
load_dotenv()  
token_required_agents = []
supervisor = Supervisor(token_required_agents=token_required_agents)
# Make sure you change the config before the configure method
supervisor.config['GMAIL_AGENT']['verbose'] = True
supervisor.config['BROWSER_AGENT']['verbose'] = True

# Configure the supervisor for automation task
supervisor.configure()
supervisor.run(query=prompt2)

print(supervisor.get_results())

