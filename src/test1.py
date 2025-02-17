from computer_interact.supervisor import Supervisor
from dotenv import load_dotenv

load_dotenv()  

query1 = "Open a web browser and naviagate to scholar.google.com"
query2 = "Search for 'OpenAI' in the search bar"
query3 = "go to gmail using google API and read the email titles 'test'"
query4 = "Open the settings in the os and set volume to zero"
query5 = "Open a firefox web browser and  type sportsbet.com using and press enter"
query6 = "Open a firefox web browser and  type scholar.google.com and enter and then search for 'OpenAI'"

""""
Prompt Design:

The best prompt will be if you specify each actino step by step. 
For example, if you need to open a web browser and navigate to scholar.google.com and search for openai, 
you can specify each step as follows:

action 1: open the  firfox web browser
action 2: type scholar.google.com 
action 3: press enter for search
action 4: type openai in the search box of google scholar
action 5: press enter for search
"""

query7 = """
action 1: open the firfox web browser
action 1.1: click on the address bar
action 2: type scholar.google.com 
action 3: press enter for search
action 4: type openai in the search box of google scholar
action 5: press enter for search
action 6: click on the address bar
action 7: press ctrl and a
action 7.1:press ctrl and c
action 8: scroll down 
action 9: click on the next button 
action 10: click on the address bar
action 11: press ctrl and a
action 12: press ctrl and c
action 13: close the browser
"""

"""
Things to do: 
1. make sure the action is complete by comparing the previous screenshot and current screenshot

"""


user_query = query6

supervisor = Supervisor()
supervisor.configure()
supervisor.run(user_query)
#print(supervisor.state)

