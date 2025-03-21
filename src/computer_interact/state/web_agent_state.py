from typing import TypedDict, Annotated, List, Literal
from operator import add
import json

class WebAgentState(TypedDict):
    user_query: str
    sender: Annotated[List[str], add]
    selected_function: str
    action_to_be_taken : str
    actions_taken : Annotated[List[str], add]
    visited_urls : Annotated[List[str], add]
    conversation_history: Annotated[List[str], add]
    new_page: Literal[True, False]
    final_answer: str
    number_of_urls_visited: int
    collect_more_info: Literal[True, False]