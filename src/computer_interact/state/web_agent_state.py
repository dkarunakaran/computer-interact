from playwright.async_api import Page, Locator
from typing import TypedDict, Annotated, List, Literal
from operator import add

class Action(TypedDict):
    thought : str
    action_type : Literal["click", "type", "scroll_read", "close_page", "wait", "go_back", "go_to_search", "retry", "respond"]
    args : str 

class WebAgentState(TypedDict):
    input: str
    page : Page
    action : Action
    actions_taken : Annotated[List[str], add]
    visited_urls : Annotated[List[str], add]
    conversation_history: Annotated[List[str], add]
    new_page: Literal[True, False]
    final_answer: str
    number_of_urls_visited: int
    collect_more_info: Literal[True, False]