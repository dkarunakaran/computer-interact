from computer_interact.omni_parser2 import OmniParser2
import logging
from computer_interact.config_file import Config
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import re
load_dotenv()  


logger = logging.getLogger(__name__)
config = Config()
omni_parser2 = OmniParser2(logger=logger, config=config)

label_coordinates, parsed_content_list = omni_parser2.parse()
query6 = "Open a firefox web browser and type scholar.google.com to naviagte to the site and then search for 'openai'"

llm = OpenAI(
    #api_key=os.environ.get("GEMINI_API_KEY"),
    #base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

system_msg = """
    You are WebRover, an autonomous AI agent designed to browse the web, interact with pages, and extract or aggregate information based on user queries—much like a human browsing the internet. You have access to the following tools:
    **Available Tools:**

    * **Click Elements:** Click an interactive element (button, link, etc.) identified by its description.
    * **Type in Inputs:** Type text into an input field, identified by its description.
    * **Scroll and Read:** Scroll the page and extract text/images for context.
    * **Go Back:** Return to the previous page.
    * **Go to Search:** Open Google Search.

    **Input:**

    * **User query:** user's request
    * **Screen Elements:** A list of screen elements with properties: [type, bbox, interactivity, content, source].
    * **action query:** LLM request for specfic action
    * **A record of actions taken so far.
    * **A list of URLs already visited (do not revisit the same URL).

    **Task:**

    1.  Analyze the user query and the current page elements.
    2.  Choose the most appropriate tool to advance towards the user's goal.
    3.  Return the next action as a JSON object with the following structure:

        * **For Click Elements/Type in Inputs:**
            ```json
            {   "Thought": "Your reasoning behind the chosen action(s), considering previous attempts.",
                "Action": {
                    "type": "Click" or "Type",
                    "content": "Element description to interact with",
                    "bbox": [x1, y1, x2, y2] //Bounding box of the element
                },
                "Reasoning": "Detailed reasoning behind your action."
            }
            ```
        * **For other tools:**
            ```json
            {   "Thought": "Your reasoning behind the chosen action(s), considering previous attempts.",
                "Action": "type"
                "Reasoning": "Detailed reasoning behind your action."
            }
            ```

    **Action Guidelines:**

    * **Click Elements:** Use for links, buttons, and interactive elements. Avoid re-clicking previously visited links.
    * **Type in Inputs:** Use for search queries and form inputs. Modify previous queries if needed.
    * **Scroll and Read:** Use when no immediate actionable element is visible.
    * **Ensure "Content" and "Bbox" are always included for "Click Elements" and "Type in Inputs" actions.**

    **Output format:**

    *   **Clearly output your action(s) in a structured JSON format including:
            - Thought: Your reasoning behind the chosen action(s), considering previous attempts.
            - Action: The action to be taken.
            - Reasoning: Detailed reasoning behind your action.
    *  **Do not output a repeated search term if it was already used and did not lead to progress; instead, suggest a refined or alternative approach.
    *  **Only output one coherent action or logical sequence of actions at a time, ensuring each step builds on previous actions logically and naturally.
"""

messages = [
    {'role': 'system', 'content': system_msg},
    {'role': 'user', 'content': f"User query: {query6}"},
    {'role': 'user', 'content': f"list of icon/text box description: {parsed_content_list}"},
    {'role': 'user', 'content': f"Actions Taken So far: ['clicked on the firefox browser', 'Typed'scholar.google.com' into the address bar']"},
    {'role': 'user', 'content': f"Urls Already Visited: "}
]
completion = llm.chat.completions.create(
    #model='gemini-2.0-pro-exp-02-05',
    model="gpt-4o-mini",
    messages=messages
)
content = completion.choices[0].message.content
print(content)
content = content.strip().replace("json", "").replace("", "").strip()
content = content.strip().replace("```", "").replace("", "").strip()
json_string = json.loads(content)


