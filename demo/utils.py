import streamlit as st
from pipeline import generator_list, retriever_list
import os
import base64
import copy
from prompts import *

def set_chat_message_style():
    st.markdown(
            """
        <style>
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

def show_file(file_path: str, type: str, role: str="user"):
    left_co, cent_co, last_co = st.columns(3)
    column = last_co if role == "user" else left_co
    with column:
        if type == "image":
            st.image(file_path)
        elif type == "audio":
            st.audio(file_path)
        else:
            st.markdown(file_path)
                
def display_chat_messages():
        
    for msg in st.session_state.messages_frontend:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        show_file(item["image_url"]["url"], "image", msg["role"])
                    elif item.get("type") == "audio_url":
                        show_file(item["audio_url"]["url"], "audio", msg["role"])
                    else:
                        st.markdown(item["text"])
            else:
                st.markdown(msg["content"])
            if msg.get("sources"):
                cols = st.columns(3)
                for i, source in enumerate(msg["sources"]):
                    with cols[i % 3].expander(f"[{i+1}] **{source['title']}**"):
                        st.markdown(source["text"])


def mode_check(generator_name: str, retriever_name: str = None) -> tuple[bool, str]:
    """Check if the selected mode matches required components and return status with message
    
    Args:
        generator_name: Name of selected generator
        retriever_name: Name of selected retriever (optional)
        
    Returns:
        tuple: (is_valid, message)
            - is_valid (bool): Whether the mode requirements are satisfied
            - message (str): Error message if requirements not met, empty if valid
    """
    # Get mode selection from sidebar
    mode = st.sidebar.selectbox(
        "选择模式",
        ["Generation", "RAG", "Agentic-Search", "MM"],
        index=0
    )

    if mode == "Generation":
        not_allowed_models = ["Search-R1-3B", "WebThinker"]
        if generator_name in not_allowed_models:
            st.sidebar.error("Generation模式不支持Agentic-Search模型")
            return False, "Generation模式不支持Agentic-Search模型"
        # Generation mode only needs a generator
        if generator_name is None:
            st.sidebar.error("Generation模式需要选择一个generator")
            return False, "Generation模式需要选择一个generator"
        return mode, ""
        
    elif mode == "RAG":
        # RAG mode needs both generator and retriever
        if generator_name is None:
            st.sidebar.error("RAG模式需要选择generator")
            return False, "RAG模式需要选择generator"
        if retriever_name is None:
            st.sidebar.error("RAG模式需要选择retriever")
            return False, "RAG模式需要选择retriever"
        return mode, ""
        
    elif mode == "Agentic-Search":
        if retriever_name is None:
            st.sidebar.error("Agentic-Search模式需要选择retriever")
            return False, "Agentic-Search模式需要选择retriever"
        # Agentic-Search mode needs specific models
        allowed_models = ["Search-R1-3B", "WebThinker"]
        if generator_name not in allowed_models:
            error_msg = f"Agentic-Search模式仅支持以下模型: {', '.join(allowed_models)}"
            st.sidebar.error(error_msg)
            return False, error_msg
        return mode, ""
        
    elif mode == "MM":
        # MM mode needs specific models
        allowed_models = ["Qwen2.5-Omni-7B"]
        if generator_name not in allowed_models:
            error_msg = f"MM模式仅支持以下模型: {', '.join(allowed_models)}"
            st.sidebar.error(error_msg)
            return False, error_msg
        return mode, ""
        
    return False, "未知的模式"



def show_options():
    with st.sidebar:
        st.header("选项")
        use_generator = st.checkbox("Generator", value=True)
        generator_name = st.selectbox("选择 Generator", options=list(generator_list.keys()))
        generator = generator_list[generator_name]

        use_retriever = st.checkbox("Retriever", value=False)
        if use_retriever:
            retriever_name = st.selectbox("选择 Retriever", options=list(retriever_list.keys()))
            retriever = retriever_list[retriever_name]
        else:
            retriever = None
            retriever_name = None

    return use_generator, generator, generator_name, use_retriever, retriever, retriever_name

def show_file_uploader():
    with st.sidebar:
        uploaded_file = st.file_uploader("上传文件", type=["txt", "png", "wav"])
        
        if uploaded_file is not None:
            os.makedirs("data/tmp", exist_ok=True)
            with open(f"data/tmp/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
        if uploaded_file is None:
            file_path = None
        else:
            file_path = f"data/tmp/{uploaded_file.name}"
        return file_path

def file_type_check(file_path: str):
    if file_path.endswith(".png"):
        return "image_url"
    elif file_path.endswith(".wav"):
        return "audio_url"
    else:
        return "text"


def encode_file(file_path: str):
    with open(file_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")
        return base64_data

def frontend_history_manager(history, prompt: str, model_name: str, file_path: str=None):
    if model_name == "Qwen2.5-Omni-7B" and file_path:
        file_type = file_type_check(file_path)
        if file_type in ["image_url", "audio_url"]:
            content = [
                {"type": file_type, f"{file_type}": {"url": file_path}},
                {"type": "text", "text": prompt if prompt else ""}
            ]
            history.append({"role": "user", "content": content})
        else:
            raise ValueError("文件类型不支持")
    else:
        if prompt:
            history.append({"role": "user", "content": prompt})


def backend_history_manager(
    history, 
    generator_name: str, 
    prompt: str, 
    mode: str, 
    evidence_str: str=None,
    file_path: str=None, 
    only_keep_last: bool=True):
    # Process prompt with prompt_manager
    processed_prompt = prompt_manager(prompt, mode, generator_name, evidence_str)

    if only_keep_last and file_path:
        # Get current file type
        current_type = file_type_check(file_path)
        
        # Remove previous entries of same type
        new_history = []
        for m in history:
            if isinstance(m["content"], list):
                content = []
                for item in m["content"]:
                    # Keep items of different types
                    if item.get("type") != current_type:
                        content.append(item)
                if content:
                    new_history.append({"role": m["role"], "content": content})
            else:
                new_history.append(m)
        
        history.clear()
        history.extend(new_history)

        # Add current file
        if current_type in ["image_url", "audio_url"]:
            # Encode file if needed
            if not file_path.startswith(("http://", "https://")):
                file_base64 = encode_file(file_path)
                encoded_url = f"data:{current_type.split('_')[0]};base64,{file_base64}"
            else:
                encoded_url = file_path
                
            content = [
                {"type": current_type, current_type: {"url": encoded_url}},
                {"type": "text", "text": processed_prompt if processed_prompt else ""}
            ]
            history.append({"role": "user", "content": content})
    else:
        # Just append the processed prompt
        if processed_prompt:
            history.append({"role": "user", "content": processed_prompt})


def prompt_manager(prompt: str, mode: str, model_name: str, evidence_str: str=None):

    if mode == "Agentic-Search":
        if model_name == "Search-R1-3B":
            return SEARCH_R1_PROMPT.format(question=prompt)
        else:
            return prompt
    elif mode == "RAG":
        return RAG_PROMPT.format(question=prompt, context=evidence_str)
    else:
        return prompt

def generate_manager(backend_history, frontend_history, generator, generator_name: str, prompt: str, mode: str, file_path: str=None):
    backend_history_manager(backend_history, generator_name, prompt, mode, file_path)
    # 流式输出
    response = ""
    response_placeholder = st.empty()
    for chunk in generator.stream_completion(backend_history):
        response += chunk
        response_placeholder.markdown(response + "▌")
    response_placeholder.markdown(response)
    frontend_history.append({"role": "assistant", "content": response})
    backend_history.append({"role": "assistant", "content": response})
    st.rerun()

def omnigen_generate_manager(frontend_history, prompt: str, generator, mode: str, file_path: str=None):
    output = generator.generate(prompt)
    frontend_history.append({"role": "assistant", "content": [output]})
    # with st.chat_message("assistant"):
    show_file(output["image_url"]["url"], "image", "assistant")
    st.rerun()

def rag_manager(frontend_history, backend_history, generator, retriever, generator_name: str, query: str, mode: str, file_path: str=None):
    
    sources = []
    evidence_list = retriever.search(query)
    evidence_str = ""
    for i, evidence in enumerate(evidence_list):
        content = evidence['document']['contents']
        title, text = content.split("\n")[0], content.split("\n")[1]
        title = title.replace("\"", "")
        sources.append({"title": title, "text": text})
        evidence_str += f"Information [{i+1}] {title}\n{text}\n"

        

    backend_history_manager(backend_history, generator_name, query, mode, evidence_str, file_path)
    response = ""
    response_placeholder = st.empty()
    for chunk in generator.stream_completion(backend_history):
        response += chunk
        response_placeholder.markdown(response + "▌")
    response_placeholder.markdown(response)
    
    frontend_history.append({"role": "assistant", "content": response, "sources": sources})
    backend_history.append({"role": "assistant", "content": response})
    st.rerun()


def agentic_search_manager(frontend_history, generator, retriever, generator_name: str, query: str, mode: str, max_turns: int=6):
    query = prompt_manager(query, mode, generator_name)
    tokenizer = generator.tokenizer
    agent = generator.generator
    special_tokens = generator.special_tokens

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}], add_generation_prompt=True, tokenize=False)
    target_sequences = [special_tokens["search_end"], special_tokens["answer_end"]]

    if generator_name == "Search-R1-3B":
        response = search_r1_generate(agent, prompt, target_sequences, retriever, special_tokens, max_turns)
    else:
        raise ValueError("不支持的模型")
    
        
    frontend_history.append({"role": "assistant", "content": response})
    st.rerun()

def search_r1_generate(agent, prompt, target_sequences, retriever, special_tokens, max_turns: int=6):
    response = ""
    response_placeholder = st.empty()
    turn = 0
    while turn < max_turns: 
        for chunk in agent.stream_plain_completion(prompt, stop=target_sequences):
            response += chunk
            response_placeholder.markdown(response + "▌")
        if special_tokens["answer_start"] in response:
            response += special_tokens["answer_end"]
            break
        sub_query = response.split(special_tokens["search_start"])[1]
        evidence_list = retriever.search(sub_query)
        evidence_str = ""
        for i, evidence in enumerate(evidence_list):
            content = evidence['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            evidence_str += f"Doc {i+1}(Title: {title}) {text}\n"

        response = f"{response}{special_tokens['search_end']}<information>{evidence_str}</information>"
        prompt += f"\n\n{response}\n\n"
        turn += 1
    return response

