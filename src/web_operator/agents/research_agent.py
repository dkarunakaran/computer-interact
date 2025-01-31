from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_openai import ChatOpenAI
#from langchain_together import ChatTogether
from web_operator.agent_tools.custom_research_toolkit import CustomResearchToolkit



class ResearchAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        llm = ChatOpenAI(model=self.cfg['model'], temperature=0) 
        #llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
        researchToolkit = CustomResearchToolkit()
        tools = researchToolkit.get_tools()
        self.context = """
            You are a research assistant. 
            you can search and retrieve the relavent papers based on user query from arxiv and pubmed
            Then provide what is the next step. 
            You have no access to extenal websites, if they want to naviagte, just give next step and ask supervisor to navigate to the site
            """ +  """
                ONLY respond to the part of query relevant to your purpose.
                IGNORE tasks you can't complete. 
        """
        #DO NOT ASK for further assistance unless user specified.
        #DO NOR REPEAT the same search.
        self.prompt = hub.pull("dkarunakaran/openai-tools-agent-with-context") 
        agent = create_openai_tools_agent(llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=self.cfg['research_agent']['verbose'])