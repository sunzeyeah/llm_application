## Features

基于[LangChain](https://github.com/hwchase17/langchain) ，实现LLM的各类应用，目前提供以下功能：

- 搜索引擎：默认使用Google
- 文本摘要：对外部文件生成摘要
- 问答机器人：基于本地知识库实现

## Usage
支持4种方式初始化langchain LLM对象：
1. OpenAI API：基于OpenAI的API
   - prerequisite-1：需要```OPENAI_API_KEY```，可在[OpenAI官网](https://platform.openai.com/account/api-keys) 申请
   - prerequisite-2：OpenAI账户为付费帐户
2. Huggingface API：基于Huggingface Inference API
   - prerequisite-1：需要```HUGGINGFACEHUB_API_TOKEN```，可在[hugginface官网](https://huggingface.co/settings/tokens) 申请
   - prerequisite-2：模型已托管在huggingface
3. Custom API：基于自定义API
   - prerequisite：已搭建好自定义API
4. Local：本地加载模型

### 1. 搜索引擎
实现搜索引擎的功能。默认使用Google，需要在[SerpApi官网](https://serpapi.com/) 申请```SERPAPI_API_KEY```
```bash
PROMPT="In what year was the film Departed with Leopnardo Dicaprio released?"
# 基于OpenAI API
python src/apps.py \
  --task "google_search" \
  --mode "openai_api" \
  --api_key $OPENAI_API_KEY \
  --serp_api_key $SERPAPI_API_KEY \
  --prompt $PROMPT
```

### 2. 文本摘要
对外部文件生成相应的摘要。```INPUT_FILE```为需要生成摘要的外部文件
```bash
# 基于OpenAI API
python src/apps.py \
  --task "summarization" \
  --mode "openai_api" \
  --api_key $OPENAI_API_KEY \
  --input_file $INPUT_FILE
```

### 3. 问答机器人
基于本地知识库，构建问答机器人。 首先，需要使用embedding工具将知识库转换为向量，默认使用```langchain.embeddings.OpenAIEmbeddings```。然后，基于向量匹配查询后进行回答。

```DATA_DIR```为本地知识库的文件地址。```VECTOR_DIR```为知识库的向量文件地址，第一次计算后向量结果即会保存在该地址，后续可直接加载，无需重复计算。
```bash
PROMPT="科大讯飞今年第一季度收入是多少？"
# 基于OpenAI API
python src/apps.py \
  --task "chatbot" \
  --mode "openai_api" \
  --api_key $OPENAI_API_KEY \
  --vector_dir $VECTOR_DIR \
  --data_dir $DATA_DIR \
  --pattern "**/*.txt" \
  --prompt $PROMPT
```

## Result
以下为OpenAI API的结果
### 1. 搜索引擎
```bash
> Entering new  chain...
 I need to find information about the movie
Action: Search
Action Input: "The Departed year released"
Observation: October 6, 2006
Thought: I now know the final answer
Final Answer: The Departed was released in 2006.

> Finished chain.

The Departed was released in 2006.
```

### 2. 文本摘要
```bash
> Entering new  chain...
Prompt after formatting:
Write a concise summary of the following:

"美国财政部长耶伦（Janet Yellen）刚刚结束了四天对华访问，她此行的目的是要重建两国之间的桥梁。

这次北京之行是否成功？要是我们只看一项最基本的指标，是的。

美国与中国再次实现了对话，面对面的，哪怕不算热情，也算是彬彬有礼和互相尊重。

这跟特朗普（Donald Trump）执政年间，大多依靠社交媒体作为喊话筒的跨太平洋沟通比较，可谓大相径庭。

双方的语调和内容都比之前正面、慎重。在耶伦此行之前，尚有美国国务卿布林肯（Antony Blinken）6月份那次事关重大的访问，两国当时均承诺要稳定彼此关系。

星期天（7月9日）行程结束之际，耶伦表示，建立“与中国新经济团队之间适应性强且有建设性的沟通渠道”将会有所帮助。此言不能被低估。

今年3月，中国政府顶层的大部分人被替换，新人的首要条件是忠诚于最高领导人习近平，而其中的关键人物这一正是该国新的经济事务主管官员何立峰。

星期六（8日），耶伦大部分时间都花在与何立峰会谈上。她说两人的会谈“直接、实质和卓有成效”，但同时承认双方有“显著分歧”。"


CONCISE SUMMARY:

US Treasury Secretary Janet Yellen recently concluded a four-day visit to China with the purpose of rebuilding bridges between the two countries. The talks were successful in terms of basic metrics, with polite and respectful communication between the two sides. Saturday's talks between Yellen and He Lifeng, the new economic affairs official, were described as "direct, substantive and productive" despite significant differences.
```


## Reference
- [LangChain官方文档](https://python.langchain.com/docs/get_started/introduction.html)
- [LangChain-Chinese-Getting-Started-Guide](https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide/tree/main)