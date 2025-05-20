# 🔍 Awesome Agentic Search

🤖 Agentic search refers to an advanced AI approach where autonomous agents plan and execute multi-step search processes to address complex queries. Unlike traditional search methods that provide static results, agentic search involves dynamic workflows where the AI agent decomposes a query, conducts iterative searches, evaluates information relevance, and synthesizes findings into coherent responses. This methodology enhances deep research capabilities, supports search-enhanced reasoning, and enables interleaved workflows, positioning AI agents as active researchers rather than passive information retrievers.


> 🚧 **Note**: This project is under intensive development and will be rapidly updated with new content, features and resources. We welcome you to join our community - please feel free to open issues, submit PRs, leave comments and ⭐ star the repo to show your support! Together we can build a comprehensive resource for advancing agentic search.

## 📚 Papers
For each paper, we provide the following information:
 > 👨‍🎓 **First Author** · 📧 **Corresponding Author (Last Author if not specified)** · 🏛️ **First Organization** · 📊 **Dataset** 
 
  *Note: Please submit a PR if we missed anything!*

📊 Dataset Types:

**General QA**: NQ, TriviaQA, PopQA

**Multi-Hop QA**: HotpotQA, 2wiki, Musique, Bamboogle

**Complex Task**: GPQA, GAIA, WebWalker QA, Humanity's Last Exam (HLE)

**Report Generation**: Glaive

**Math & Coding**: AIME, MATH500, AMC, LiveCodeBench

### 🎓 Training-based
[Search-R1: Training LLMs to Reason and Leverage Search
Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1?style=social)

 > 👨‍🎓 **Bowen Jin** · 📧 **Jiawei Han** · 🏛️ **UIUC** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-3B / 7B · 🎯 **Training**: GRPO, PPO


[WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/abs/2504.21776):![GitHub Repo stars](https://img.shields.io/github/stars/RUC-NLPIR/WebThinker?style=social)

 > 👨‍🎓 **Xiaoxi Li** · 📧 **Zhicheng Dou** · 🏛️ **GSAI, RUC** \
 > 📊 **Dataset**: Complex Task, Report Generation · 🤖 **Model**: QwQ 32B · 🎯 **Training**: SFT, DPO


[DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) [![[code]](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)](https://github.com/GAIR-NLP/DeepResearcher)

 > 👨‍🎓 **Yuxiang Zheng** · 📧 **Pengfei Liu** · 🏛️ **SJTU** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-7B · 🎯 **Training**: GRPO


[R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592) [![[code]](https://img.shields.io/github/stars/RUCAIBox/R1-Searcher)](https://github.com/RUCAIBox/R1-Searcher)

 > 👨‍🎓 **Huatong Song** · 📧 **Wayne Xin Zhao** · 🏛️ **GSAI, RUC** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-7B, Llama-3.1-8B · 🎯 **Training**: SFT, GRPO, REINFORCE++

[ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588) [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/ZeroSearch)](https://github.com/Alibaba-NLP/ZeroSearch)

 > 👨‍🎓 **Hao Sun** · 📧 **Zile Qiao, Jiayan Guo, Yan Zhang** · 🏛️ **Tongyi Lab** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-3B / 7B, s LLaMA-3.2-3B · 🎯 **Training**: REINFORCE, GRPO, PPO

[Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342) [![[code]](https://img.shields.io/github/stars/microsoft/LMOps)](https://github.com/microsoft/LMOps/)

 > 👨‍🎓 **Liang Wang** · 📧 **Furu Wei** · 🏛️ **MSRA** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Llama-3.1-8B-Instruct · 🎯 **Training**: REINFORCE, GRPO, PPO

[IKEA: Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent](https://arxiv.org/abs/2505.07596) [![[code]](https://img.shields.io/github/stars/hzy312/knowledge-r1)](https://github.com/hzy312/knowledge-r1)

 > 👨‍🎓 **Ziyang Huang** · 📧 **Kang Liu** · 🏛️ **IA, CAS** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-3B / 7B · 🎯 **Training**: GRPO

[Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging](https://arxiv.org/abs/2505.09316)

> 👨‍🎓 **Hongjin Qian** · 📧 **Zheng Liu** · 🏛️ **BAAI** \
> 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-3B / 7B · 🎯 **Training**: GRPO, PPO


[Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs](https://arxiv.org/pdf/2505.11277) [![[code]](https://img.shields.io/github/stars/syr-cn/AutoRefine)](https://github.com/syr-cn/AutoRefine)

> 👨‍🎓 **Yaorui Shi** · 📧 **Xiang Wang** · 🏛️ **USTC** \
> 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-3B · 🎯 **Training**: GRPO



### 🔄 Workflow-based
[Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366):  [![[code]](https://img.shields.io/github/stars/sunnynexus/Search-o1?style=social)](https://github.com/sunnynexus/Search-o1)

 > 👨‍🎓 **Xiaoxi Li** · 📧 **Zhicheng Dou** · 🏛️ **GSAI, RUC** \
 > 📊 **Dataset**: General QA, Multi-Hop QA, Complex Task, Math & Coding · 🤖 **Model**: QwQ-32B-Preview

 [Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research](https://arxiv.org/pdf/2502.04644) [![[code]](https://img.shields.io/github/stars/theworldofagents/Agentic-Reasoning)](https://github.com/theworldofagents/Agentic-Reasoning)

> 👨‍🎓 **Junde Wu** · 📧 **Yuyuan Liu** · 🏛️ **Oxford University** \
> 📊 **Dataset**: Complex Task · 🤖 **Model**: APIs


###  🔧 Tool Using
[OTC: Optimal Tool Calls via Reinforcement Learning](https://arxiv.org/pdf/2504.14870)

 > 👨‍🎓 **Hongru Wang** · 📧 **Heng Ji** · 🏛️ **CUHK** \
 > 📊 **Dataset**: General QA, Multi-Hop QA, Math & Coding · 🤖 **Model**: Qwen-2.5-3B / 7B· 🎯 **Training**: GRPO, PPO

### 🖼️ Multi-Modal
[Multimodal-Search-R1: Incentivizing LMMs to Search](https://kimingng.notion.site/MMSearch-R1-Incentivizing-LMMs-to-Search-1bcce992031880b2bc64fde13ef83e2a) [![[code]](https://img.shields.io/github/stars/EvolvingLMMs-Lab/multimodal-search-r1)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)

 > 👨‍🎓 **Jinming Wu** · 📧 **Zejun Ma** · 🏛️ **BUPT** \
 > 📊 **Dataset**: VQA · 🤖 **Model**: Qwen2.5-VL-Instruct-3B/7B · 🎯 **Training**: GRPO

<!-- ### 📊 Evaluation and Dataset -->

### 🏢 Industry Solutions

OpenAI's  Deep Research: https://openai.com/index/introducing-deep-research/

Google's Gemini Pro: https://www.google.com/search/about/

X's Grok 3: https://x.ai/news/grok-3

Perplexity: https://www.perplexity.ai/

Jina AI: https://jina.ai/deepsearch/

Metasota: https://metaso.cn/

## 📝 Slides
We maintain a collection of 📊 paper presentation slides on Overleaf to facilitate learning and knowledge sharing in the agentic search community. Each presentation consists of 3-5 slides that concisely introduce key aspects of a paper, including motivation, methodology, and main results. These slides serve as quick references for understanding important works in the field and can be used for self-study, teaching, or research presentations.

🔗 Check out our slides collection: [Agentic Search Paper Slides](https://www.overleaf.com/read/dhbksrdxswps#3990a3)

## 📊 Evaluation

We maintain a comprehensive evaluation table comparing different agentic search methods across key metrics like search efficiency, result quality, and reasoning capabilities. The table will be regularly updated as new methods and benchmarks emerge to help researchers track progress in the field.

## 🏆 Arena
We are building an arena page to benchmark different agentic search methods using standardized retrieval and web browser interfaces. This will enable fair comparisons between various approaches and help advance the state-of-the-art in agentic search.

## 🤝 Contributing
We welcome contributions to this repository! If you have any suggestions or feedback, please feel free to open an issue or submit a pull request.


## 📖 Citation
If you find this repository useful, please consider citing it as follows:
```bibtex
@misc{awesome-agentic-search,
  author = {Hongjin Qian, Zheng Liu},
  title = {Awesome Agentic Search},
  year = {2025},
  publisher = {GitHub},
```