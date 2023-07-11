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
默认使用Google，需要在[SerpApi官网](https://serpapi.com/) 申请```SERPAPI_API_KEY```
```bash
# 基于OpenAI API
python src/apps.py \
  --mode "openai_api" \
  --task "google_search" \
  --serp_api_key $SERPAPI_API_KEY
```

### 2. 文本摘要
默认使用Google，需要在[SerpApi官网](https://serpapi.com/) 申请```SERPAPI_API_KEY```
```bash
# 基于OpenAI API
python src/apps.py \
  --mode "openai_api" \
  --task "google_search" \
  --serp_api_key $SERPAPI_API_KEY
```

### 3. 问答机器人
默认使用Google，需要在[SerpApi官网](https://serpapi.com/) 申请```SERPAPI_API_KEY```
```bash
# 基于OpenAI API
python src/apps.py \
  --mode "openai_api" \
  --task "google_search" \
  --serp_api_key $SERPAPI_API_KEY
```


## Reference
- [LangChain官方文档](https://python.langchain.com/docs/get_started/introduction.html)
- [LangChain-Chinese-Getting-Started-Guide](https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide/tree/main)