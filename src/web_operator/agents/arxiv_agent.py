from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_openai import ChatOpenAI

class ArxivAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        llm = ChatOpenAI(model=self.cfg['arxiv_agent']['model'], temperature=0) 
        tools = load_tools(
            ["arxiv"],
        )
        self.context = """
            You are a arxiv assistant. 
            you can search and retrieve the relavent papers based on user query from arxiv.org
            """ +  """
                ONLY respond to the part of query relevant to your purpose.
                IGNORE tasks you can't complete. 
        """
        #DO NOT ASK for further assistance unless user specified.
        #DO NOR REPEAT the same search.
        self.prompt = hub.pull("dkarunakaran/openai-tools-agent-with-context") 
        agent = create_openai_tools_agent(llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=self.cfg['arxiv_agent']['verbose'])