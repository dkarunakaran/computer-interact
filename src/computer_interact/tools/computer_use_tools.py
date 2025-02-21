from langchain_core.tools import tool
from pydantic import BaseModel, Field

class ClickInput(BaseModel):
    x: int = Field(description="x poisiton of the item you need to click")
    y: int = Field(description="y poistion of the item you need to click")
    desc: str = Field(description="A description of what you want to click on")

@tool("click-tool", args_schema=ClickInput, return_direct=True)
def click(x: int, y: int, desc: str) -> bool:
    """Click on an item on the screen."""
    return True





tools = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click on an item on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A description of what you want to click on"
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            },
            "strict": True
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_url",
            "description": "type the url on the address bar of the browser",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A description of what you want to type in the address bar of the browser"
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "press a keyboard key",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Name of the key you want to press"
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]
