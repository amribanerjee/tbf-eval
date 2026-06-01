1. arxiv:2310.06770 | SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
True Authors: Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan (Princeton NLP).

The Technical Reality: Introduces a testbed of 2,294 engineering problems from 12 open-source Python repositories (e.g., scikit-learn, matplotlib, sympy). Evaluation relies entirely on applying an LLM-generated patch file to a repository environment and executing its unit test suite to yield a binary $1$ (Resolved) or $0$ (Unresolved).

Your Citation Justification: This serves as the primary evaluation standard and source dataset for your agent trajectories. You cite it to anchor the current paradigm of automated, execution-based evaluation, paving the way for your argument that while outcome validation is objective, it hides structural flaws in the trajectory itself.

2. arxiv:2405.15793 | SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
True Authors: John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, Ofir Press (Princeton NLP).

The Technical Reality: This is the foundational paper that establishes SWE-agent and introduces the concept of the Agent-Computer Interface (ACI). It highlights how tailored, streamlined commands (for viewing, searching, and chunk-editing files) alongside defensive environmental feedback dramatically reduce the compound action error rates of language models.

Your Citation Justification: You will use their publicly released SWE-agent execution logs as part of your primary dataset. You cite it in Section 2 and 4 to illustrate how structural variations in an ACI shape the resulting step trajectories, providing the raw material that Trajectory Behavioral Fingerprinting (TBF) converts into behavioral attributions.

3. arxiv:2506.02064 | The Measurement Imbalance in Agentic AI Evaluation Undermines Industry Productivity Claims
True Authors: Kiana Jafari Meimandi, Gabriela Aránguiz-Dias, Grace Ra Kim, Lana Saadeddin, Allie Griffith, Mykel J. Kochenderfer (Stanford University).  

The Technical Reality: A systematic review evaluating 84 papers published between 2023 and 2025. It proves that current evaluation practices exhibit a dangerous technical bias, where technical metrics dominate 83% of assessments, while safety (53%), human-centered (30%), and economic dimensions (30%) are isolated on the periphery. It presents clear case studies from industries like healthcare showing how systems excelling purely on baseline technical metrics break catastrophically upon real deployment due to unmeasured context and drift.  

Your Citation Justification: This is the core validation for your Introduction (Section 1) and Discussion (Section 6). You cite it to directly justify the critical need for process metrology over outcome metrics, anchoring your framework as a technical instrument to bridge the "measurement gap" they expose. 

4. arxiv:2308.03688 | AgentBench: Evaluating LLMs as Agents
True Authors: Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, et al. (Tsinghua, ICLR 2024).  

The Technical Reality: Introduces a multi-dimensional, multi-turn evolving benchmark spanning 8 distinct interactive environments (such as Operating Systems, Databases, Web Shopping, and Digital Card Games) designed to assess LLM reasoning and decision-making capabilities. It highlights how poor long-term reasoning and multi-turn instruction following act as the core failure points for open-source configurations compared to commercial counterparts.  

Your Citation Justification: Sourced as a fundamental survey for your Related Work (Section 2). You cite it to show that while the agent community has built multi-turn interactive testbeds across diverse domains, evaluation remains tethered to task completion or episodic goal metrics rather than checking the aggregate process consistency of the generated step sequences.
