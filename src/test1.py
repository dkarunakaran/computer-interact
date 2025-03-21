from computer_interact.supervisor import Supervisor
from dotenv import load_dotenv

load_dotenv()  

query1 = "Open a web browser and naviagate to scholar.google.com"
query2 = "Search for 'OpenAI' in the search bar"
query3 = "go to gmail using google API and read the email titles 'test'"
query4 = "Open the settings in the os and set volume to zero"
query5 = "Open a firefox web browser and  type sportsbet.com using and press enter"
query6 = "Open a firefox web browser and  type scholar.google.com and enter and then search for 'OpenAI'. No need to go to the next page"

"""
Things to do: 
1. make sure the action is complete by comparing the previous screenshot and current screenshot

"""


user_query = query6

supervisor = Supervisor()

print(supervisor.config)
supervisor.configure()
supervisor.run(user_query)
#print(supervisor.state)

