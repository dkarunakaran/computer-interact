from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import sqlite3

def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

class SQLiteAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        llm = ChatOpenAI(model=self.cfg['model'], temperature=0) 
        engine = get_engine_for_chinook_db()
        db = SQLDatabase(engine)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        self.context = """
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct query to run, then look at the results of the query and return the answer.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the below tools. Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
            To start you should ALWAYS look at the tables in the database to see what you can query.
            Do NOT skip this step.
            Then you should query the schema of the most relevant tables.
        """
        #DO NOT ASK for further assistance unless user specified.
        #DO NOR REPEAT the same search.
        self.prompt = hub.pull("dkarunakaran/openai-tools-agent-with-context") 
        agent = create_openai_tools_agent(llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=self.cfg['sqlite_agent']['verbose'])