import streamlit as st
st.set_page_config(page_title="🧠 Agentic Chat Demo", page_icon="🧠", layout="wide")


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
        with st.chat_message("assistant"):
            st.markdown("_RAG Pipeline 暂未实现_")
            raise ValueError("RAG Pipeline 暂未实现")
    elif use_generator:
        # 使用 generator agent 流式输出
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response = ""
            # 构造历史消息用于多轮对话
            backend_history_manager(st.session_state.messages_backend, generator_name, prompt, mode, file_path)
            # 流式输出
            for chunk in generator.stream_completion(st.session_state.messages_backend):
                response += chunk
                response_placeholder.markdown(response + "▌")
            response_placeholder.markdown(response)
            st.session_state.messages_frontend.append({"role": "assistant", "content": response})
            st.session_state.messages_backend.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        with st.chat_message("assistant"):
            st.markdown("_请至少选择 Generator_")
            raise ValueError("请至少选择 Generator")


