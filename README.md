# <div align="center">🔍 Awesome Agentic Search</div>
<h4 align="center">
<p>
<a href="#-objectives">🎯 Objectives</a> |
<a href="#-papers">📚 Papers</a> |
<a href="#-slides">📊 Slides</a> |
<a href="#-demo">🎮 Demo</a> |
<a href="#-arena">🏆 Arena</a> |
<a href="#-gym">🏋️ Gym</a>
</p>
</h4>
🤖 Agentic search is an advanced AI approach where autonomous agents actively plan and execute multi-step, iterative searches to decompose complex queries, evaluate relevance, and synthesize responses—transforming them from passive retrievers into dynamic, reasoning-driven researchers.

## 🎯 Objectives
> 🚧 **Note**: This project is evolving rapidly—join the community by opening issues, submitting PRs, leaving comments, or  ⭐ starring the repo to help build a leading resource for agentic search.

- **Research Collection**: Curate and categorize comprehensive research work in agentic search, including papers, code implementations, and empirical findings

- **Interactive Demos**: Build demonstration pages to showcase different agentic search methods and allow hands-on exploration of their capabilities

- **Evaluation Arena**: Develop a Python toolkit for systematic evaluation and benchmarking of agentic search methods across diverse tasks and metrics

- **Training Gym**: Create a Python framework for training and optimizing agentic search models, including reinforcement learning and other approaches


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

[An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents](https://arxiv.org/pdf/2505.15117) ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1?style=social)

 > 👨‍🎓 **Bowen Jin** · 📧 **Jiawei Han** · 🏛️ **UIUC** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-3B / 7B /  14B· 🎯 **Training**: GRPO, PPO
 
 *Notes: a new version of Search-R1.*

[WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/abs/2504.21776):![GitHub Repo stars](https://img.shields.io/github/stars/RUC-NLPIR/WebThinker?style=social)

 > 👨‍🎓 **Xiaoxi Li** · 📧 **Zhicheng Dou** · 🏛️ **GSAI, RUC** \
 > 📊 **Dataset**: Complex Task, Report Generation · 🤖 **Model**: QwQ 32B · 🎯 **Training**: SFT, DPO


[DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) [![[code]](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)](https://github.com/GAIR-NLP/DeepResearcher)

 > 👨‍🎓 **Yuxiang Zheng** · 📧 **Pengfei Liu** · 🏛️ **SJTU** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-7B · 🎯 **Training**: GRPO


[R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592) [![[code]](https://img.shields.io/github/stars/RUCAIBox/R1-Searcher)](https://github.com/RUCAIBox/R1-Searcher)

 > 👨‍🎓 **Huatong Song** · 📧 **Wayne Xin Zhao** · 🏛️ **GSAI, RUC** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-7B, Llama-3.1-8B · 🎯 **Training**: SFT, GRPO, REINFORCE++

 [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.17005) [![[code]](https://img.shields.io/github/stars/RUCAIBox/R1-Searcher-plus)](https://github.com/RUCAIBox/R1-Searcher-plus)

  > 👨‍🎓 **Huatong Song** · 📧 **Wayne Xin Zhao** · 🏛️ **GSAI, RUC** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-7B · 🎯 **Training**: SFT, GRPO, REINFORCE++

[SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis](https://arxiv.org/abs/2505.16834) [![[code]](https://img.shields.io/github/stars/RUCAIBox/SimpleDeepSearcher)](https://github.com/RUCAIBox/SimpleDeepSearcher)

  > 👨‍🎓 **Shuang Sun** · 📧 **Wayne Xin Zhao** · 🏛️ **GSAI, RUC** \
 > 📊 **Dataset**: General QA, Multi-Hop QA · 🤖 **Model**: Qwen-2.5-7B / 32B, QwQ-32B · 🎯 **Training**: SFT, DPO, REINFORCE++

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

[ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.15776) [![[code]](https://img.shields.io/github/stars/BeastyZ/ConvSearch-R1)](https://github.com/BeastyZ/ConvSearch-R1)

> 👨‍🎓 **Changtai Zhu** · 📧 **Xipeng Qiu** · 🏛️ **FDU** \
> 📊 **Dataset**: Conversational QA · 🤖 **Model**: Qwen-2.5-3B / Llama-3.2-3B · 🎯 **Training**: SFT, GRPO




### 🔄 Workflow-based
[Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366):  [![[code]](https://img.shields.io/github/stars/sunnynexus/Search-o1?style=social)](https://github.com/sunnynexus/Search-o1)

 > 👨‍🎓 **Xiaoxi Li** · 📧 **Zhicheng Dou** · 🏛️ **GSAI, RUC** \
 > 📊 **Dataset**: General QA, Multi-Hop QA, Complex Task, Math & Coding · 🤖 **Model**: QwQ-32B-Preview

 [Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research](https://arxiv.org/pdf/2502.04644) [![[code]](https://img.shields.io/github/stars/theworldofagents/Agentic-Reasoning)](https://github.com/theworldofagents/Agentic-Reasoning)

> 👨‍🎓 **Junde Wu** · 📧 **Yuyuan Liu** · 🏛️ **Oxford University** \
> 📊 **Dataset**: Complex Task · 🤖 **Model**: APIs


###  🔧 Tool Using

[Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning](https://arxiv.org/abs/2505.16410) [![[code]](https://img.shields.io/github/stars/dongguanting/Tool-Star)](https://github.com/dongguanting/Tool-Star)

> 👨‍🎓 **Guanting Dong** · 📧 **Zhicheng Dou** · 🏛️ **GSAI, RUC** \
> 📊 **Dataset**: General QA, Multi-Hop QA, Math & Coding · 🤖 **Model**: Qwen-2.5-3B· 🎯 **Training**: SFT,GRPO, PPO

[OTC: Optimal Tool Calls via Reinforcement Learning](https://arxiv.org/pdf/2504.14870)

 > 👨‍🎓 **Hongru Wang** · 📧 **Heng Ji** · 🏛️ **CUHK** \
 > 📊 **Dataset**: General QA, Multi-Hop QA, Math & Coding · 🤖 **Model**: Qwen-2.5-3B / 7B· 🎯 **Training**: GRPO, PPO

### 🖼️ Multi-Modal
[Multimodal-Search-R1: Incentivizing LMMs to Search](https://kimingng.notion.site/MMSearch-R1-Incentivizing-LMMs-to-Search-1bcce992031880b2bc64fde13ef83e2a) [![[code]](https://img.shields.io/github/stars/EvolvingLMMs-Lab/multimodal-search-r1)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)

 > 👨‍🎓 **Jinming Wu** · 📧 **Zejun Ma** · 🏛️ **BUPT** \
 > 📊 **Dataset**: VQA · 🤖 **Model**: Qwen2.5-VL-Instruct-3B/7B · 🎯 **Training**: GRPO

### 📊 Evaluation and Dataset

[InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation](https://arxiv.org/abs/2505.15872) [![[code]](https://img.shields.io/github/stars/YunjiaXi/InfoDeepSeek)](https://github.com/YunjiaXi/InfoDeepSeek)

> 👨‍🎓 **Yunjia Xi** · 📧 **Jianghao Lin** · 🏛️ **SJTU** \
> 📊 **Dataset**: General QA, Multi-Hop QA

[BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents](https://arxiv.org/abs/2504.12516) [![[code]](https://img.shields.io/github/stars/openai/simple-evals)](https://github.com/openai/simple-evals)

> 👨‍🎓 **Jason Wei** · 📧 **Amelia Glaese** · 🏛️ **OpenAI** \
> 📊 **Dataset**: Web Browsing


### 🏢 Industry Solutions

OpenAI's  Deep Research: https://openai.com/index/introducing-deep-research/

Google's Gemini Pro: https://www.google.com/search/about/

X's Grok 3: https://x.ai/news/grok-3

Perplexity: https://www.perplexity.ai/

Jina AI: https://jina.ai/deepsearch/

Metasota: https://metaso.cn/

## 🎮 Demo
We are building a demo page to showcase different agentic search methods and allow hands-on exploration of their capabilities. Each demo will be integrated into a standardized retrieval and web browser interface with comparable settings, enabling comprehensive and fair comparisons across various approaches. This systematic evaluation will help identify strengths and limitations of different methods and advance the state-of-the-art in agentic search.

## 📝 Slides
We maintain a collection of 📊 paper presentation slides on Overleaf to facilitate learning and knowledge sharing in the agentic search community. Each presentation consists of 3-5 slides that concisely introduce key aspects of a paper, including motivation, methodology, and main results. These slides serve as quick references for understanding important works in the field and can be used for self-study, teaching, or research presentations.

🔗 Check out our slides collection: [Agentic Search Paper Slides](https://www.overleaf.com/read/dhbksrdxswps#3990a3)

## 🏆 Arena
We are building an arena page to benchmark different agentic search methods in a unified evaluation framework. All methods will be integrated into standardized retrieval and web browser interfaces with comparable settings, enabling comprehensive and fair comparisons across various approaches. This systematic evaluation will help identify strengths and limitations of different methods and advance the state-of-the-art in agentic search.

## 🏋️ Gym
We are organizing a collection of optimization frameworks and training approaches used in agentic search, including reinforcement learning methods like GRPO and PPO, as well as supervised fine-tuning techniques. This will help researchers understand and implement effective training strategies for their agentic search models.

Stay tuned for detailed tutorials and code examples on training agentic search systems!



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