from agent.api_agent import Agent
import streamlit as st
import json
import requests
import time
import transformers
import torch
torch.classes.__path__ = []
class omnigen_agent:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def notify(self, message):
        
        _placeholder = st.empty()
        info = ""
        for char in message:
            info += char
            _placeholder.markdown(info + "▌")
            time.sleep(0.05)
        _placeholder.markdown(info)

    def generate(self, prompt):
        self.notify("正在生成图片，这可能需要一些时间...")
        # Make API request
        try:
            response = requests.post(
                self.api_url,
                json={"prompt": prompt},
                headers={"Content-Type": "application/json"}
            )
            
            result = response.json()
            
            if result.get("success"):
                return {"type": "image_url", "image_url": {"url": result["image_path"]}}
            else:
                return {"type": "text", "text": f"生成失败: {result.get('error', '未知错误')}"}
                
        except Exception as e:
            return {"type": "text", "text": f"请求失败: {str(e)}"}

class wiki_retriever:
    def __init__(self, url: str):
        self.search_url = url

    def search(self, query: str, topk: int = 3):
        try:
            # Prepare payload
            payload = {
                "queries": [query],
                "topk": topk,
                "return_scores": True
            }
            
            # Make API request
            response = requests.post(self.search_url, json=payload)
            response.raise_for_status()
            
            # Parse results
            results = response.json()
            if results and "result" in results:
                # Return first batch of results since we only have one query
                return results["result"][0]
            return []
            
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return []
 

class agentic_searcher:
    def __init__(self, 
    model: str, 
    model_path: str, 
    special_tokens: dict):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.generator = Agent(
                model=model,
                source="vllm",
                base_url=api_config["vllm"][model]["base_url"],
            )
        self.special_tokens = special_tokens

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
def get_agentic_searcher(model: str):
    model_path = api_config["vllm"][model]["model_path"]
    special_tokens = api_config["vllm"][model]["special_tokens"]
    return agentic_searcher(model, model_path, special_tokens)

@st.cache_resource
def get_wikipedia_retriever():
    return None

@st.cache_resource
def get_web_retriever():
    return None

with open("config/api_config.json", "r") as f:
    api_config = json.load(f)

generator_list = {
    "Qwen2.5-Omni-7B": get_vllm_agent("Qwen2.5-Omni-7B"),
    "Search-R1-3B": get_agentic_searcher("SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo"),
    "omnigen-v2": omnigen_agent(api_config["omnigen"]["base_url"]),
    "gpt-4o-mini": get_openrouter_agent("gpt-4o-mini"),
    "gpt-4o": get_openrouter_agent("gpt-4o"),
    "deepseek-chat": get_deepseek_chat("deepseek-chat"),   
}

retriever_list = {
    "Wikipedia": wiki_retriever(api_config["wiki_retriever"]["base_url"]),
}


