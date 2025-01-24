from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
import os
from agent_tools.custom_gmail_toolkit import GmailToolkit


class GmailAgent:
    def __init__(self, logger):
        self.logger = logger
        self.context = ""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        gmail_toolkit = GmailToolkit()
        self.tools = gmail_toolkit.get_tools()
        self.prompt = hub.pull("ebahr/openai-tools-agent-with-context:3b3e6baf") 
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, prompt):
        return self.agent_executor.invoke({"chat_history":[], "agent_scratchpad":"", "context": self.context,"input": prompt})