FROM python:3.12.5-bookworm

RUN pip install --upgrade pip

# Google API
RUN pip install google-api-python-client==2.159.0
RUN pip install google-auth==2.37.0
RUN pip install google-auth-httplib2==0.2.0
RUN pip install google-auth-oauthlib==1.2.0

# Web scraping
RUN pip install beautifulsoup4~=4.12.3
RUN pip install lxml==5.3.0
RUN pip install PyYAML==6.0.2


# Langchain, Langgraph, GenAI
RUN pip install langchain==0.3.14
RUN pip install langchain-chroma==0.2.0
RUN pip install langchain-ollama==0.2.2
RUN pip install langchain-huggingface==0.1.2
RUN pip install langchain_community==0.3.14
RUN pip install langchain-openai==0.3.0
RUN pip install langgraph==0.2.60 
RUN pip install langgraph-checkpoint-sqlite==2.0.1
RUN pip install -q -U google-generativeai==0.8.4
RUN pip install langchain-google-community==2.0.4

# Browser
RUN pip install pytest-playwright==0.6.2
RUN playwright install
RUN apt-get update
RUN playwright install-deps 
RUN playwright install chrome

# Web interface
RUN pip install streamlit==1.41.1

# Other
RUN pip install pandas==2.2.3
RUN pip install python-dotenv==1.0.1

WORKDIR /app

EXPOSE 7000

#For testing
CMD ["/bin/bash"]

# For production
#CMD ["python", "app.py"]
