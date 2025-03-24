tools = [
    {
        "type": "function",
        "function": {
            "name": "os_agent",
            "description": "This is the operating system agent. This agent is helpful when we need to develop computer use capability specifically for automating the OS tasks.",
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
            "name": "web_agent",
            "description": "This is the web agent where we need to do the search, browse, or naviagte using the browser",
            "parameters": {  
                "type": "object",
                "properties": {}, 
                "additionalProperties": False 
            },
            "strict": True
        }
    }
]