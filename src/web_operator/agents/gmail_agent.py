from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from web_operator.agent_tools.custom_gmail_toolkit import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
import os


class GmailAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        creds = get_gmail_credentials(
            token_file=os.environ.get("GOOGLE_API_TOKEN_LOC"),
            scopes=["https://mail.google.com/"],
            client_secrets_file=os.environ.get("GOOGLE_API_CREDS_LOC"),
        )
        api_resource = build_resource_service(credentials=creds)
        gmail_toolkit = GmailToolkit(api_resource=api_resource)
        tools = gmail_toolkit.get_tools()
        self.context = """
            You are a gmail assistant. you can perform various operation in gmail such as reading, deleting, and so operation in gmail using gmail tools.
            Use your tools to answer questions. Do not send or delete email at any corcumstances unless users has asked to do. If you do not have a tool to answer the question, say so.
        """ +  """
                ONLY respond to the part of query relevant to your purpose.
                IGNORE tasks you can't complete. 
        """
        self.prompt = hub.pull("dkarunakaran/openai-tools-agent-with-context") 
        agent = create_openai_tools_agent(llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=self.cfg['GMAIL_AGENT']['verbose'])
