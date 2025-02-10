tools = [
    {
        "type": "function",
        "function": {
            "name": "computer_use_node",
            "description": "This is the computer automation node using LLM. This node is helpful when we don't have APIs of requested services and need to go through the browser.",
            "parameters": {  
                "type": "object",
                "properties": {}, 
                "additionalProperties": False 
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "api_operation_node",
            "description": "This node is activated when we can do operation using APIs of different services.",
            "parameters": {  
                "type": "object",
                "properties": {}, 
                "additionalProperties": False 
            },
            "strict": True
        }
    }
]