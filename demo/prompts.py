SEARCH_R1_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

RAG_PROMPT = """Answer the question based on the retrieved information. If the question is in a specific language, provide the answer in that same language. First analyze the retrieved information, then provide a direct and concise answer based on the relevant information. Make sure to cite the relevant information sources using [1], [2], etc. after each statement that uses that information. If the information is insufficient to answer the question, state that clearly.

Question: {question}

Retrieved information:
{context}
"""

