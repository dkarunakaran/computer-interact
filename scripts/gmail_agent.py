from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
import os
from agent_tools.custom_gmail_toolkit import GmailToolkit


class GmailAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        gmail_toolkit = GmailToolkit()
        tools = gmail_toolkit.get_tools()
        self.context = """
            You are a gmail assistant. you can perform various operation in gmail such as reading, deleting, and so operation in gmail using gmail tools.
            Use your tools to answer questions. If you do not have a tool to answer the question, say so.
        """ +  """
                ONLY respond to the part of query relevant to your purpose.
                IGNORE tasks you can't complete. 
        """
        self.prompt = hub.pull("ebahr/openai-tools-agent-with-context:3b3e6baf") 
        agent = create_openai_tools_agent(llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=self.cfg['GMAIL_AGENT']['verbose'])
