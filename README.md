# ERNIEBot Researcher

选题调研的Web所下图所示，用户输入关键词或者自然的语句，后台根据给定的文献进行搜索相关内容，然后使用文心大模型生成研究报告。

<div align="center">
    <img src="https://github.com/w5688414/ERNIEBot-Researcher/assets/12107462/d4f9100a-fa29-4912-9f7a-be151a7f5ee2" width="800px">
</div>

生成的报告下载地址：[报告下载](https://github.com/w5688414/ERNIEBot-Researcher/files/14901642/polish__research_report.pdf)

<div align="center">
    <img src="https://github.com/w5688414/ERNIEBot-Researcher/assets/12107462/6e7ee373-ae45-4bc4-b177-69fc6fc7dad1" width="500px">
</div>

ERNIEBot Researcher是一个自主智能体（Autonomous Agent），旨在对各种任务进行全面的在线研究。能够精心编撰内容详尽、真实可信且立场公正的中文研究报告，同时根据需求提供针对特定资源、结构化大纲以及宝贵经验教训的深度定制服务。汲取了近期备受瞩目的[Plan-and-Solve](https://arxiv.org/abs/2305.04091)技术的精髓，并结合当前流行的[RAG](https://arxiv.org/abs/2005.11401)技术的优势，ERNIEBot Researcher通过多Agent协作和高效并行处理机制，有效攻克了速度瓶颈、决策确定性及成果可靠性等难题。

### 为什么需要ERNIEBot Researcher？

+ 手动研究任务形成客观结论可能需要时间，有时需要数周才能找到正确的资源和信息。
+ 目前的LLM是根据过去和过时的信息进行训练的，产生幻觉的风险很大，这使得产生的报告几乎与研究任务无关。
+ LLM生成的报告一般没有做段落级/句子级别的文献来源引用，生成的内容无法进行追踪和验证。


## News and Updates

+ 2024.04.04 发布ERNIEBot Researcher，支持erniebot和chatgpt完成研究任务，支持OpenAI Embedding，ERNIE-Embedding。


## 架构

主要思想是运行“planner”和“execution” agents，而planner生成问题进行研究，execution agents根据每个生成的研究问题寻求最相关的信息。最后，planner 过滤并汇总所有相关信息，并创建一份研究报告。

Agents利用ernie-4.0和ernie-longtext来完成研究任务， ernie-4.0主要用于做决策和规划，ernie-longtext主要用于撰写报告。


<div align="center">
    <img src="https://github.com/PaddlePaddle/ERNIE-SDK/assets/12107462/2cedc93d-0482-44bd-ba30-4e5697e8a559" width="700px">
</div>

## 应用特色

+ 基于研究查询或任务创建特定领域的Agent。
+ 根据现有知识库的内容生成一组多样性的研究问题，这些问题共同形成对任何给定任务的客观意见。
+ 对于每个研究问题，从知识库中选择与给定问题相关的信息。
+ 过滤和汇总所有的信息来源，并生成最终的研究报告。
+ 多个报告Agent并行生成，并保持一定的多样性。
+ 使用思维链技术对多个报告进行质量评估和排序，克服伪随机性，并选择最优的报告。
+ 使用反思机制对报告进行修订和完善。
+ 使用检索增强和chain of verification对事实进行校验。
+ 使用润色机制提升报告的整体可读性，融合更多的细节描述。

**注意**
1. 生成一次报告需要花费10min以上，并且research agent设置的越多，消耗的时间越长，同时会消耗大量的Tokens。
2. 报告生成的质量与输入给应用的文档的质量有关，适合网页，期刊，企业办公文档等场景，不适合文本比较少，无用信息过多的文档报告生成场景。

## 快速开始

> 第一步：下载项目源代码

```
git https://github.com/PaddlePaddle/ERNIE-SDK.git
cd ernie-agent/applications/erniebot_researcher
```

> 第二步：安装依赖

```
pip install -r requirements.txt
```
如果上面的命令失败，请运行下面的命令：

```
conda create -n researcher39 -y python=3.9 && conda activate researcher39
pip install -r requirements.txt
```
源码安装ernie-agent:

```
cd ernie-agent
pip install -e .
```

> 第三步：下载中文字体

```
wget https://paddlenlp.bj.bcebos.com/pipelines/fonts/SimSun.ttf
```

> 第四步：构建文档索引


支持azure openai_embedding和ernie_embedding两种向量类型，其中ernie-embedding需要在[AI Studio星河社区](https://aistudio.baidu.com/index)注册并登录账号，然后在AI Studio的[访问令牌页面](https://aistudio.baidu.com/index/accessToken)获取`Access Token`，最后设置环境变量:

```
export EB_AGENT_ACCESS_TOKEN=<aistudio-access-token>
export AISTUDIO_ACCESS_TOKEN=<aistudio-access-token>
export EB_AGENT_LOGGING_LEVEL=INFO
```
对于azure OpenAI embeding则需要设置openai相关的环境变量：

```
export AZURE_OPENAI_ENDPOINT="<your azure openai endpoint>"
export AZURE_OPENAI_API_KEY="<your azure openai api key>"
```
我们支持docx、pdf、txt等格式的文件，用户可以把这些文件放到同一个文件夹下，然后运行下面的命令创建索引，后续会根据这些文件写报告。

为了方便测试，我们提供了样例数据。 样例数据：

```
wget https://paddlenlp.bj.bcebos.com/pipelines/erniebot_researcher_example.tar.gz
tar xvf erniebot_researcher_example.tar.gz
```
url数据：如果用户有文件对应的url链接，可以传入存储url链接的txt。在txt中，每一行存储url链接和对应文件的路径，例如:

```
https://zhuanlan.zhihu.com/p/659457816 erniebot_researcher_example/Ai_Agent的起源.md
```

如果用户不传入url文件，则默认文件的路径为其url链接

摘要数据：用户可以利用path_abstract参数传入自己文件对应摘要的存储路径。 其中摘要需要用json文件存储。其中json文件内存储的是多个字典，每个字典有3组键值对，
- `page_content` : `str`, 文件摘要。
- `url` : `str`, 文件url链接。
- `name` : `str`, 文件名字。

例如：

```
[{"page_content":"文件摘要","url":"https://zhuanlan.zhihu.com/p/659457816","name":Ai_Agent的起源},
...]
```

如果用户没有摘要路径，则无需改变path_abstract的默认值，我们会利用ernie-4.0来自动生成摘要，生成的摘要存储路径为abstract.json。

接下来运行：

```
python ./tools/preprocessing.py \
--index_name_full_text <the index name of your full text> \
--index_name_abstract <the index name of your abstract text> \
--path_full_text <the folder path of your full text> \
--url_path <the path of your url text> \
--path_abstract <the json path of your abstract text>
```

> 第五步：运行

```
python demo.py --num_research_agent 1 \
                                --index_name_full_text <your full text> \
                                --index_name_abstract <your abstract text>
```
- `index_name_full_text`: 全文知识库索引的路径
- `index_name_abstract`: 摘要知识库索引的路径
- `index_name_citation`: 参考文献的索引路径
- `num_research_agent`: 生成报告的agent数量
- `iterations`: 反思迭代的轮数
- `chatbot`: LLM的类型，目前支持erniebot和chatgpt
- `report_type`: 报告的类型，目前支持research_report
- `embedding_type`: 使用的向量类型，目前支持ernie_embedding和openai_embedding(azure)
- `save_path`:报告保存的路径
- `server_name`: webui的ip地址
- `server_port`: webui的端口号
- `log_path`: 日志的保存路径
- `use_ui`: 是否使用webui
- `use_reflection`: 是否使用反思过程
- `fact_checking`:是否使用事实性校验过程
- `framework`: 基于的框架，目前支持langchain

## Reference

[1] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, Ee-Peng Lim:
[Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091). ACL (1) 2023: 2609-2634

[2] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren:
[Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents](https://arxiv.org/abs/2304.09542). EMNLP 2023: 14918-14937


## :heart: Acknowledge
我们借鉴了 Assaf Elovic [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 优秀的框架设计，在此对[GPT Researcher](https://github.com/assafelovic/gpt-researcher)作者及其开源社区表示感谢。

We learn form the excellent framework design of Assaf Elovic [GPT Researcher](https://github.com/assafelovic/gpt-researcher), and we would like to express our thanks to the authors of GPT Researcher and their open source community.
