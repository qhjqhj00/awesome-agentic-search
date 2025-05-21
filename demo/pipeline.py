from agent.api_agent import Agent
import streamlit as st
import json

with open("config/api_config.json", "r") as f:
    api_config = json.load(f)

@st.cache_resource
def get_openrouter_agent(model):
    return Agent(
        model=model,
        source="openrouter",
        base_url=api_config["openrouter"]["base_url"],
        api_key=api_config["openrouter"]["api_key"]
    )

@st.cache_resource
def get_deepseek_chat(model):
    return Agent(
        model=model,
        source="deepseek",
        base_url=api_config["deepseek"]["base_url"],
        api_key=api_config["deepseek"]["api_key"]
    )

@st.cache_resource
def get_deepseek_reasoner(model):
    return Agent(
        model=model,
        source="deepseek",
        base_url=api_config["deepseek"]["base_url"],
        api_key=api_config["deepseek"]["api_key"]
    )

@st.cache_resource
def get_vllm_agent(model):
    return Agent(
        model=model,
        source="vllm",
        base_url=api_config["vllm"][model]["base_url"],
    )

@st.cache_resource
def get_wikipedia_retriever():
    return None

@st.cache_resource
def get_web_retriever():
    return None

generator_list = {
    "Qwen2.5-Omni-7B": get_vllm_agent("Qwen2.5-Omni-7B"),
    "Search-R1-3B": get_vllm_agent("SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo"),
    "gpt-4o-mini": get_openrouter_agent("gpt-4o-mini"),
    "gpt-4o": get_openrouter_agent("gpt-4o"),
    "deepseek-chat": get_deepseek_chat("deepseek-chat"),   
}

retriever_list = {
    "Wikipedia": get_wikipedia_retriever(),
    "Web Search": get_web_retriever(),
}


