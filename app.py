import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

# ✅ Fix the deprecated import
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  

import os
from dotenv import load_dotenv

# ✅ Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ✅ Initialize conversation state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        # ✅ Ensure API key is provided
        if not api_key:
            st.warning("⚠️ Please enter your Groq API Key in the sidebar.")
        else:
            llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
            tools = [search, arxiv, wiki]

            # ✅ Fix: Pass only `prompt` instead of full `messages` (avoids `results` issue)
            search_agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = search_agent.run(prompt, callbacks=[st_cb])  # ✅ Fix: Use `prompt`, not full history
                st.session_state.messages.append({'role': 'assistant', "content": response})
                st.write(response)

    except Exception as e:
        st.error(f"❌ Error: {e}")  # ✅ Catch and display errors properly


