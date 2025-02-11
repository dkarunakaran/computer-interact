from web_operator.supervisor import Supervisor
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
action 2: type scholar.google.com 
action 3: press enter for search
action 4: type openai in the search box of google scholar
action 5: press enter for search
action 6: move mouse to an empty space on the page
action 7: do the right click
action 8: select the save as option
action 9: type page1 as the name for the page
action 10: save the page
action 11: scroll the page down
action 12: click the next button
action 13: select the save as option
action 14: type page2 as the name for the page
action 15: save the page
action 16: close the browser
"""

user_query = query7

supervisor = Supervisor()
supervisor.configure()
supervisor.run(user_query)


