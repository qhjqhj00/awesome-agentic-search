import streamlit as st
st.set_page_config(page_title="Agentic Chat Demo", page_icon="🧠", layout="wide")


from agent.api_agent import Agent
import json

from utils import *
import copy
with open("config/api_config.json", "r") as f:
    api_config = json.load(f)

# Streamlit 页面设置

st.title("🧠 Agentic Chat Demo")
set_chat_message_style()
use_generator, generator, generator_name, use_retriever, retriever, retriever_name = show_options()

if "messages_frontend" not in st.session_state:
    st.session_state.messages_frontend = []
if "messages_backend" not in st.session_state:
    st.session_state.messages_backend = []

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


if prompt := st.chat_input("请输入你的问题..."):
   
    with st.chat_message("user"):
        if file_path:
            file_type = file_type_check(file_path)
            show_file(file_path, file_type.split("_")[0])
        st.markdown(prompt)
    

    frontend_history_manager(st.session_state.messages_frontend, prompt, generator_name, file_path)
    # 选择调用对象
    if use_generator and use_retriever:
        # 预留 rag_pipeline
        if mode == "RAG":
            rag_manager(st.session_state.messages_frontend, st.session_state.messages_backend, generator, retriever, generator_name, prompt, mode, file_path)
        elif mode == "Agentic-Search":
            agentic_search_manager(st.session_state.messages_frontend, generator, retriever, generator_name, prompt, mode)
        else:
            raise ValueError("Pipeline 暂未实现")
    elif use_generator:
        # 使用 generator agent 流式输出
        with st.chat_message("assistant"):
            # 构造历史消息用于多轮对话
            if generator_name == "omnigen-v2":
                omnigen_generate_manager(st.session_state.messages_frontend, prompt, generator, mode, file_path)
            else:
                generate_manager(st.session_state.messages_backend, st.session_state.messages_frontend, generator, generator_name, prompt, mode, file_path)
    else:
        with st.chat_message("assistant"):
            st.markdown("_请至少选择 Generator_")
            raise ValueError("请至少选择 Generator")


