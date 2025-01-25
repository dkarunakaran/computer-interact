from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    message: Annotated[List[str], operator.add]
    sender: Annotated[List[str], operator.add]