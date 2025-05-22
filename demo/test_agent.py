from agent.api_agent import Agent
import json
import transformers
def get_vllm_agent(model):
    return Agent(
        model=model,
        source="vllm",
        base_url=api_config["vllm"][model]["base_url"],
    )




with open("config/api_config.json", "r") as f:
    api_config = json.load(f)

agent = get_vllm_agent("SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo")


model_id = "/share/qhj/dw/LLMs/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

query = "which city is beside the capital of France?"
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {query}\n"""


doc = "The capital of France is Paris."
target_sequences = ["</search>"]

prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)


response = ""
for chunk in agent.stream_plain_completion(prompt, stop=target_sequences):
    response += chunk
    print(chunk, end="", flush=True)

prompt += f"\n\n{response}</search><information>{doc}</information>\n\n"
print("\n\n")
print("=="*10)
response = ""
for chunk in agent.stream_plain_completion(prompt, stop=target_sequences):
    response += chunk
    print(chunk, end="", flush=True)


