import os
from typing import Any

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils import logger
from src.tasks.base import Task

# refine prompt template (used only for chain_type=refine)
REFINE_PROMPT_TMPL_EN = """Your job is to produce a final summary. We have provided an existing summary: {existing_answer}


We have the opportunity to refine the existing summary (only if needed) with some the following context: {text}


Given the new context, refine the original summary. If the context isn't useful, return the original summary."""
REFINE_PROMPT_EN = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=REFINE_PROMPT_TMPL_EN,
)
REFINE_PROMPT_TMPL_ZH = """任务描述：根据现有摘要和上下文信息，生成最终的摘要。


现有摘要：{existing_answer}


上下文信息：{text}


请根据上下文信息，对已有摘要进行修改。如果该上下文信息没有用, 请返回原摘要。"""
REFINE_PROMPT_ZH = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=REFINE_PROMPT_TMPL_ZH,
)


PROMPT_TEMPLATE_EN = """Write a concise summary of the following:

{text}

CONCISE SUMMARY:"""
PROMPT_EN = PromptTemplate(template=PROMPT_TEMPLATE_EN, input_variables=["text"])
PROMPT_TEMPLATE_ZH = """任务描述：根据提供的文本，生成对应摘要。

文本：{text}

摘要："""
PROMPT_ZH = PromptTemplate(template=PROMPT_TEMPLATE_ZH, input_variables=["text"])


class Summarization(Task):

    def __init__(self,
                 chain_type: str = "refine",
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.chain_type = chain_type
        self.prompt = PROMPT_ZH if self.language == "zh" else PROMPT_EN
        if chain_type == "refine":
            self.refine_prompt = REFINE_PROMPT_ZH if self.language == "zh" else REFINE_PROMPT_EN

    # def _init_tools(self, **kwargs: Any) -> None:
    #     """Initialize Tools"""
    #     tools = kwargs.get("tools", [])
    #     self.tool_names = tools
    #     self.tools = load_tools(tools)
    #
    # def _init_agent(self, agent: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #                 verbose: bool = True,
    #                 **kwargs: Any) -> None:
    #     """Initialize Agent"""
    #     self.agent = initialize_agent(self.tools, self.llm, agent=agent, verbose=verbose)

    @property
    def _task_name(self) -> str:
        """Return name of task"""
        return "summarization"

    def __call__(self,
                 input_file: str,
                 chunk_size: int = 2048,
                 chunk_overlap: int = 0
                 ) -> str:
        # 导入文本
        loader = UnstructuredFileLoader(input_file)
        # 将文本转成 Document 对象
        document = loader.load()
        logger.info(f'document length: {len(document)}')

        # 初始化文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 切分文本
        split_documents = text_splitter.split_documents(document)
        logger.info(f'chunked document length: {len(split_documents)}')

        # 创建总结链
        if self.chain_type == "refine":
            chain = load_summarize_chain(self.llm, chain_type=self.chain_type, verbose=True,
                                         question_prompt=self.prompt, refine_prompt=self.refine_prompt)
        else:
            chain = load_summarize_chain(self.llm, chain_type=self.chain_type, verbose=True,
                                         prompt=self.prompt)

        # 执行总结链，（为了快速演示，只总结前5段）
        return chain.run(split_documents[:5])
