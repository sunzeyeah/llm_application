import os
import json
from typing import Any, Dict, Optional, List

from langchain import VectorDBQA, PromptTemplate
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.base import Embeddings
from langchain.schema import Document, BaseLanguageModel, BasePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore
from langchain.document_loaders.helpers import detect_file_encodings
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR

from src.utils import rmdir, logger
from src.tasks.base import Task


PROMPT_TEMPLATE_EN = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT_EN = PromptTemplate(
    template=PROMPT_TEMPLATE_EN, input_variables=["context", "question"]
)

PROMPT_TEMPLATE_ZH = """使用以下信息来回答问题。如果信息中不包含问题的答案，请直接说你不知道，不要编造答案。

{context}

问题：{question}
答案："""
PROMPT_ZH = PromptTemplate(
    template=PROMPT_TEMPLATE_ZH, input_variables=["context", "question"]
)

DOCUMENT_PROMPT_TEMPLATE = PromptTemplate(input_variables=["page_content", "answer"],
                                          template="{page_content}\n{answer}")


class FAQLoader(TextLoader):
    """Load Quesion-Answer jsonl files (FAQ knowledge base)
    Only use question/prompt as page_content (for retrieval), the answer/label is metadata
    """
    def load(self) -> List[Document]:
        """Load from file path."""
        docs = []
        try:
            for line in open(self.file_path, encoding=self.encoding):
                d = json.loads(line.strip("\n"))
                docs.append(Document(page_content=d['prompt'], metadata={"answer": d['label']}))
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    logger.debug("Trying encoding: ", encoding.encoding)
                    try:
                        for line in open(self.file_path, encoding=encoding.encoding):
                            d = json.loads(line.strip("\n"))
                            docs.append(Document(page_content=d['prompt'], metadata={"answer": d['label']}))
                    except UnicodeDecodeError:
                        docs = []
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        # metadata = {"source": self.file_path}
        return docs


def format_document(doc: Document, prompt: BasePromptTemplate) -> str:
    """Format a document into a string based on a prompt template.

    First, this pulls information from the document from two sources:

    1. `page_content`:
        This takes the information from the `document.page_content`
        and assigns it to a variable named `page_content`.
    2. metadata:
        This takes information from `document.metadata` and assigns
        it to variables of the same name.

    Those variables are then passed into the `prompt` to produce a formatted string.

    Args:
        doc: Document, the page_content and metadata will be used to create
            the final string.
        prompt: BasePromptTemplate, will be used to format the page_content
            and metadata into the final string.

    Returns:
        string of the document formatted.

    Example:
        .. code-block:: python

            from langchain.schema import Document
            from langchain.prompts import PromptTemplate
            doc = Document(page_content="This is a joke", metadata={"page": "1"})
            prompt = PromptTemplate.from_template("Page {page}: {page_content}")
            format_document(doc, prompt)
            >>> "Page 1: This is a joke"
    """
    base_info = {"page_content": doc.page_content, **doc.metadata}
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [
            iv for iv in prompt.input_variables if iv != "page_content"
        ]
        raise ValueError(
            f"Document prompt requires documents to have metadata variables: "
            f"{required_metadata}. Received document with missing metadata: "
            f"{list(missing_metadata)}."
        )
    document_info = {k: base_info[k] for k in prompt.input_variables}
    return prompt.format(**document_info)


class FAQDocumentsChain(StuffDocumentsChain):

    document_prompt: BasePromptTemplate = DOCUMENT_PROMPT_TEMPLATE
    document_separator: str = "\n"

    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and docs.

        Format and the join all the documents together into one input with name
        `self.document_variable_name`. The pluck any additional variables
        from **kwargs.

        Args:
            docs: List of documents to format and then join into single input
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            dictionary of inputs to LLMChain
        """
        # Format each document according to the prompt
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        # Join the documents together to put them in the prompt.
        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
        return inputs


class FAQRetrievalQA(RetrievalQA):

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: Optional[PromptTemplate] = None,
            **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        combine_documents_chain = FAQDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )

        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "faq_retrieval_qa"


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
        if not os.path.exists(vector_dir):
            os.mkdir(vector_dir)

        if data_dir is not None:
            # 删除原embedding文件
            rmdir(vector_dir)
            # 加载文件夹中的所有txt类型的文件
            loader = DirectoryLoader(data_dir, glob=pattern, loader_cls=FAQLoader, show_progress=True,
                                     use_multithreading=True, max_concurrency=8, loader_kwargs={"encoding": "utf-8"})
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
        # qa = VectorDBQA.from_chain_type(llm=self.llm, chain_type="stuff", vectorstore=self.vector_store,
        #                                 return_source_documents=True, chain_type_kwargs={"prompt": self.prompt},
        #                                 verbose=self.verbose)
        retriever = self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
        qa = FAQRetrievalQA.from_llm(llm=self.llm,  prompt=self.prompt, retriever=retriever,
                                     return_source_documents=True, verbose=self.verbose)
        # 进行问答
        return qa({"query": query})
