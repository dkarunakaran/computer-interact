
computeruse_tools = [
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