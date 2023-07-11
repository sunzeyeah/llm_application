import os
from typing import Any, Dict

from langchain import VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore

from src.utils import logger
from src.tasks.base import Task


class ChatBot(Task):

    def __init__(self,
                 embeddings: Embeddings,
                 vector_dir: str,
                 data_dir: str = None,
                 pattern: str = None,
                 chunk_size: int = 512,
                 chunk_overlap: int = 0,
                 **kwargs: Any) -> None:
        # 初始化 openai 的 embeddings 对象
        self.embeddings = embeddings
        # 初始化 vector_store
        self._init_vector_store(vector_dir, data_dir, pattern, chunk_size, chunk_overlap)
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
        return "chatbot"

    def _init_vector_store(self,
                           vector_dir: str,
                           data_dir: str = None,
                           pattern: str = None,
                           chunk_size: int = 512,
                           chunk_overlap: int = 0,) -> None:
        if data_dir is not None:
            # 加载文件夹中的所有txt类型的文件
            loader = DirectoryLoader(data_dir, glob=pattern)
            # 将数据转成 document 对象，每个文件会作为一个 document
            documents = loader.load()
            # 初始化加载器
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            # 切割加载的 document
            split_docs = text_splitter.split_documents(documents)
            # 将 document 通过 openai 的 embeddings 对象计算 embedding向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
            self.vector_store = Chroma.from_documents(split_docs, self.embeddings, persist_directory=vector_dir)
            # 持久化数据
            self.vector_store.persist()
        else:
            # 加载数据
            self.vector_store = Chroma(persist_directory=vector_dir, embedding_function=self.embeddings)

    def __call__(self,
                 query: str
                 ) -> Dict[str, Any]:
        # 创建问答对象
        qa = VectorDBQA.from_chain_type(llm=self.llm, chain_type="stuff", vectorstore=self.vector_store,
                                        return_source_documents=True)
        # 进行问答
        return qa({"query": query})
