from web_operator.supervisor import Supervisor
from web_operator.utils import google_api_authenticate
from dotenv import load_dotenv

"""
Steps:
    1. Create a google cloud account using the gmail account you want to access by navigating to https://console.cloud.google.com/apis/credentials?project=bold-gearbox-408603
    2. Created token and download as credentials.json in the local path
    3. Update the .env file with new path
    GOOGLE_API_CREDS_LOC=/media/beastan/projects/web-operator/examples/credentials.json
    4. Then run the below code, it will open a new browser tab that lead to google cloud sign in page
    load_dotenv()  
    token_required_agents = []
    supervisor = Supervisor(token_required_agents=token_required_agents)
    google_api_authenticate(supervisor.config)
    5. Then follow the step to authroize and download the token.json file
    6. It has been downloaded to the same folder as credentials.json
    7. update the .env file with token.json file location
    GOOGLE_API_TOKEN_LOC=/media/beastan/projects/web-operator/examples/token.json
    8. Now we can remove the google_api_authenticate code as we got the files for future authentication
    9. then follow the bwlow steps to proceed with automation.
"""

prompt = """
        go to gmail and find email with subject 'Open-Source Rival to OpenAI's Reasoning Model'
        We need only the content of the latest email of the above subject and disgard other emails.
        Extract the first URL (link) from the email content.
        Naviagte to the URL and summarise the content and no further navigation is required

        **Constraints:**
        - Only extract the first URL found in the email body.
        - If no URL is found, return "No URL found."

        """
load_dotenv()  
token_required_agents = ['gmail_agent']
supervisor = Supervisor(token_required_agents=token_required_agents)

# Make sure you change the config before the configure method
supervisor.config['GMAIL_AGENT']['verbose'] = True
supervisor.config['BROWSER_AGENT']['verbose'] = True

# Configure the supervisor for automation task
supervisor.configure()
supervisor.run(query=prompt)

print(supervisor.get_results())




