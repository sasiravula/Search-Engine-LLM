import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki) # Initialize the query runner

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv) # Initialize the query runner

search = DuckDuckGoSearchRun(name="Search") # Initialize the query runner

st.title("Langchain - Chat with search")
"""
In this example we are using "StreamlitCallbackHandler" to display the thoughts and actions of the agent in the Streamlit app. 
Try more langchain Streamlit agent examples at [github.com/langchain-ai/streanlit-agents]

"""

st.sidebar.title("Settings")
groq_api_key=st.sidebar.text_input("Enter your Groq API key", type='password')

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant", "content":"Hello! I am a chatbot that can search the web. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
if prompt := st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)
    tools=[search, arxiv, wiki]    
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)
    
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)
  
        