import streamlit as st
st.set_page_config(page_title="Agentic Chat Demo", page_icon="🧠", layout="wide")


from agent.api_agent import Agent
import json
from pipeline import agentic_searcher
from utils import *
import copy
with open("config/api_config.json", "r") as f:
    api_config = json.load(f)

# Streamlit page settings

st.title("🧠 Agentic Chat Demo")
set_chat_message_style()
use_generator, generator, generator_name, use_retriever, retriever, retriever_name = show_options()

if "messages_frontend" not in st.session_state:
    st.session_state.messages_frontend = []
if "messages_backend" not in st.session_state:
    st.session_state.messages_backend = []

if hasattr(generator, "special_tokens"):
    display_chat_messages(generator.special_tokens)
else:
    display_chat_messages()
# Add clear history button to sidebar
with st.sidebar:
    if st.button("Clear History"):
        st.session_state.messages_frontend = []
        st.session_state.messages_backend = []
        st.rerun()

mode, mode_message = mode_check(generator_name, retriever_name)
if not mode:
    st.error(mode_message)
    st.stop()

file_path = show_file_uploader()


if prompt := st.chat_input("Enter your question..."):
   
    with st.chat_message("user"):
        if file_path:
            file_type = file_type_check(file_path)
            show_file(file_path, file_type.split("_")[0])
        st.markdown(prompt)
    

    frontend_history_manager(st.session_state.messages_frontend, prompt, generator_name, file_path)
    # Select pipeline to call
    if use_generator and use_retriever:
        # Reserved for rag_pipeline
        if mode == "RAG":
            rag_manager(st.session_state.messages_frontend, st.session_state.messages_backend, generator, retriever, generator_name, prompt, mode, file_path)
        elif mode == "Agentic-Search":
            agentic_search_manager(st.session_state.messages_frontend, generator, retriever, generator_name, prompt, mode)
        else:
            raise ValueError("Pipeline not implemented yet")
    elif use_generator:
        # Use generator agent for streaming output
        with st.chat_message("assistant"):
            # Construct history messages for multi-turn dialogue
            if generator_name == "omnigen-v2":
                omnigen_generate_manager(st.session_state.messages_frontend, prompt, generator, mode, file_path)
            else:
                generate_manager(st.session_state.messages_backend, st.session_state.messages_frontend, generator, generator_name, prompt, mode, file_path)
    else:
        with st.chat_message("assistant"):
            st.markdown("_Please select at least one Generator_")
            raise ValueError("Please select at least one Generator")
