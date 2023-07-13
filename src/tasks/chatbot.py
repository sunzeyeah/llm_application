import os
from typing import Any, Dict

from langchain import VectorDBQA, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore

from src.utils import rmdir
from src.tasks.base import Task


PROMPT_TEMPLATE_EN = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT_EN = PromptTemplate(
    template=PROMPT_TEMPLATE_EN, input_variables=["context", "question"]
)

PROMPT_TEMPLATE_ZH = """使用以下信息来回答问题。如果你不知道答案，请直接说你不知道，不要编造答案。

{context}

问题：{question}
答案："""
PROMPT_ZH = PromptTemplate(
    template=PROMPT_TEMPLATE_ZH, input_variables=["context", "question"]
)


class ChatBot(Task):

    def __init__(self,
                 embeddings: Embeddings,
                 vector_dir: str,
                 data_dir: str = None,
                 pattern: str = None,
                 chunk_size: int = 2048,
                 chunk_overlap: int = 0,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 初始化 openai 的 embeddings 对象
        self.embeddings = embeddings
        # 初始化 vector_store
        self._init_vector_store(vector_dir, data_dir, pattern, chunk_size, chunk_overlap)
        # 初始化prompt template
        self.prompt = PROMPT_ZH if self.language == "zh" else PROMPT_EN

    @property
    def _task_name(self) -> str:
        """Return name of task"""
        return "chatbot"

    def _init_vector_store(self,
                           vector_dir: str,
                           data_dir: str,
                           pattern: str,
                           chunk_size: int,
                           chunk_overlap: int,) -> None:
        if data_dir is not None:
            # 删除原embedding文件
            rmdir(vector_dir)
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
                 query: str,
                 search_type: str = "similarity",
                 k: int = 3,
                 ) -> Dict[str, Any]:
        # 创建问答对象
        chain_type_kwargs = {"prompt": self.prompt}
        # qa = VectorDBQA.from_chain_type(llm=self.llm, chain_type="stuff", vectorstore=self.vector_store,
        #                                 return_source_documents=True, chain_type_kwargs=chain_type_kwargs,
        #                                 verbose=self.verbose)
        retriever = self.vector_store.as_retriever(search_type=search_type, k=k)
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever,
                                         chain_type_kwargs=chain_type_kwargs, verbose=self.verbose)
        # 进行问答
        return qa({"query": query})
