response_json_supervisor = {
    "type": "json_schema",
    "json_schema": {
        "name": "step_reasoning",
        "schema": {
            "type": "object",
            "properties": {
                "steps_in_details": {
                    "type": "string",
                    "description": "More detailed description of all steps in the process"  
                },
                "next_step": {
                    "type": "string",
                    "description": "The next step in the process. if it is the first step, then it should be the first step in the process."
                }
            },
            "required": ["steps_in_details", "next_step"],
            "additionalProperties": False
        },
        "strict": True
    }
}


response_json_supervisor_wo_steps_details = {
        "type": "json_schema",
        "json_schema": {
            "name": "step_reasoning",
            "schema": {
                "type": "object",
                "properties": {
                    "next_step": {
                        "type": "string",
                        "description": "The next step in the process. if it is the first step, then it should be the first step in the process."
                    }
                },
                "required": ["next_step"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

