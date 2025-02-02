from web_operator.supervisor import Supervisor
from web_operator.utils import google_api_authenticate
from dotenv import load_dotenv



next_task = """
        Your task is to extract invoice data from the email content.

        Reply Structures:
        - Amount 
        - Due_date 
        - Biller_name   

        Reply with valid json. Please make sure Due_date is in year-month-day format and Biller_name has only a few words. Please do not add list or dict as value to Biller_name. 
      """

prompt1 = f"""
        Go to gmail and read the data from  emailed titled Uniti Retail Pty Ltd - Invoice ID/Ref 701484.
        {next_task}
        then create a table called content using SQL and impor the data
        Once you done that return FINISH
        """

load_dotenv()  
required_agents = ['gmail_agent', 'sqlite_agent']
supervisor = Supervisor(required_agents=required_agents)
# Make sure you change the config before the configure method
supervisor.config['browser_agent']['headless'] = False
supervisor.config['browser_agent']['verbose'] = True

# Configure the supervisor for automation task
supervisor.configure()
supervisor.run(query=prompt1)
print(supervisor.get_results())



