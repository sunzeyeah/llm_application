import os
from typing import Any

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils import logger
from src.tasks.base import Task


class Summarization(Task):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

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
                 chunk_size: int = 512,
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
        chain = load_summarize_chain(self.llm, chain_type="refine", verbose=True)

        # 执行总结链，（为了快速演示，只总结前5段）
        return chain.run(split_documents[:5])
