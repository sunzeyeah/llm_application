import sys

sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/llm_application")
sys.path.insert(0, "/Users/zeyesun/Documents/Code/llm_application")
sys.path.insert(0, "D:\\Code\\llm_application")
import os
import shutil
import gradio as gr
import uuid
import torch
from typing import List, Dict, Tuple, Union
from gradio.inputs import File
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, VectorStore
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter

from src.utils import logger, rmdir, list_dir
from src.tasks.chatbot import FAQLoader
from src.tasks import Task
from src.apps import init_llm, init_task


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Gradio Settings
block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""
webui_title = """
# 🎉LLM Application WebUI🎉
👍 [https://github.com/sunzeyeah/llm_application](https://github.com/sunzeyeah/llm_application)
"""
default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)
flag_csv_logger = gr.CSVLogger()
FLAG_USER_NAME: str = uuid.uuid4().hex
task_list_zh = [
    "搜索引擎",
    "文本摘要",
    "知识库问答",
]
task_en_to_zh = {
    "google_search": "搜索引擎",
    "summarization": "文本摘要",
    "chatbot": "知识库问答",
}
task_zh_to_en = {
    "搜索引擎": "google_search",
    "文本摘要": "summarization",
    "知识库问答": "chatbot",
}
default_task = "chatbot"
default_llm_model = "bloomz-560M"
default_embedding_name = "text2vec-large-chinese"
default_kb_name = "test"
init_message = f"""欢迎使用 LLM Application Web UI！

请在右侧切换任务，目前支持{len(task_list_zh)}类：{" ".join([f"({i+1}) {t}" for i, t in enumerate(task_list_zh)])}

当前任务：{task_en_to_zh[default_task]}
当前LLM模型：{default_llm_model}
当前embedding模型：{default_embedding_name}
当前知识库：{default_kb_name}
"""

# LangChain and LLM Params
args = {
    "mode": "local",
    "task": default_task,
    # "model_name": f"/mnt/pa002-28359-vol543625-share/LLM/checkpoint/{default_llm_model}",
    "model_name": f"/Users/zeyesun/Documents/Data/models/{default_llm_model}",
    # "model_name": f"D:\\Data\\models\\{default_llm_model}",
    "local_rank": 0,
    "checkpoint": None,
    "bits": 16,
    "max_length_generation": 256,
    "do_sample": False,
    "top_p": 0.9,
    "temperature": 0.9,
    "repetition_penalty": 1.0,
    "chunk_size": 1024,
    "chunk_overlap": 0,
    # "vector_dir": "/mnt/pa002-28359-vol543625-private/Data/chatgpt/output/embeddings",
    # "embedding_name": f"/mnt/pa002-28359-vol543625-share/LLM/checkpoint/{default_embedding_name}",
    "vector_dir": "/Users/zeyesun/Documents/Data/chatgpt/output/embeddings",
    "embedding_name": f"/Users/zeyesun/Documents/Data/models/{default_embedding_name}",
    # "vector_dir": "D:\\Data\\chatgpt\\output\\embeddings",
    # "embedding_name": f"D:\\Data\\models\\{default_embedding_name}",
    "kb_name": default_kb_name,
    "search_type": "similarity",
    "k": 5,
    "search_threshold": 0.7,
    "serp_api_key": None,
}
args = dotdict(args)
llm_model_list = [
    "chatglm2-6B",
    "bloomz-560M",
]
embedding_model_list = [
    "text2vec-large-chinese",
]

# Global Variables
llm: LLM = None
langchain_task: Task = None
embeddings: Embeddings = None
vector_store: VectorStore = None


def init_embeddings_and_vector_store(vector_dir: str,
                                     data_dir: str = None,
                                     pattern: str = None) -> str:
    global embeddings
    global vector_store
    try:
        embedding_device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name=args.embedding_name,
                                           model_kwargs={'device': embedding_device})
        if data_dir is not None:
            # 删除原embedding文件
            rmdir(vector_dir)
            # 加载文件夹中的所有txt类型的文件
            loader = DirectoryLoader(data_dir, glob=pattern, loader_cls=FAQLoader, show_progress=True,
                                     use_multithreading=True, max_concurrency=8, loader_kwargs={"encoding": "utf-8"})
            # 将数据转成 document 对象，每个文件会作为一个 document
            documents = loader.load()
            # 初始化加载器
            text_splitter = CharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            # 切割加载的 document
            split_docs = text_splitter.split_documents(documents)
            # 将 document 通过 openai 的 embeddings 对象计算 embedding向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
            vector_store = Chroma.from_documents(split_docs, embeddings, persist_directory=vector_dir)
            # 持久化数据
            vector_store.persist()
            kb_status = f"""知识库：{os.path.basename(vector_dir)}已成功新建并加载"""
        else:
            vector_store = Chroma(persist_directory=vector_dir, embedding_function=embeddings)
            kb_status = f"""知识库：{os.path.basename(vector_dir)}已成功加载"""
    except Exception as e:
        logger.error(e)
        kb_status = f"""【WARNING】知识库：{os.path.basename(vector_dir)}加载失败"""
        logger.warning(kb_status)

    return kb_status


def initialize_llm() -> str:
    global llm
    try:
        llm = init_llm(args)
        llm_status = f"""LLM模型：{os.path.basename(args.model_name)}已成功加载"""
    except Exception as e:
        logger.error(e)
        llm_status = f"""【WARNING】LLM模型：{os.path.basename(args.model_name)}加载失败，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.warning(llm_status)

    return llm_status


def initialize_task() -> str:
    global langchain_task
    try:
        langchain_task = init_task(args, llm, embeddings)
        task_status = f"""任务：{task_en_to_zh[args.task]}已成功加载，可以开始对话"""
    except Exception as e:
        logger.error(e)
        if args.task == "google_search" and args.serp_api_key is None:
            task_status = f"""【WARNING】任务：{task_en_to_zh[args.task]}默认使用Google，需要SERP_API_KEY，请在右侧输入框内进行输入"""
        else:
            task_status = f"""【WARNING】任务：{task_en_to_zh[args.task]}加载失败"""
        logger.warning(task_status)

    return task_status


def reinit_model(llm_model: str,
                 embedding_model: str,
                 kb_name: str,
                 # llm_history_len, no_remote_model, use_ptuning_v2, use_lora,
                 history: List[List[str]]) -> List[List[str]]:
    global args
    try:
        model_dir = os.sep.join(args.model_name.split(os.sep)[:-1])
        args.model_name = os.path.join(model_dir, llm_model)
        args.embedding_name = os.path.join(model_dir, embedding_model)
        kb_status = init_embeddings_and_vector_store(vector_dir=os.path.join(args.vector_dir, kb_name))
        logger.debug(kb_status)
        initialize_llm()
        llm_status = f"""LLM模型成功更换为{llm_model}，Embedding模型成功更换为{embedding_model}，成功加载知识库：{kb_name}"""
        logger.debug(llm_status)
    except Exception as e:
        logger.error(e)
        llm_status = f"""【WARNING】LLM模型未更换为{llm_model}，Embedding模型未更换为{embedding_model}，
        请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.warning(llm_status)
    return history + [[None, llm_status]]


def get_kb_list() -> List[str]:
    try:
        kb_names = list_dir(args.vector_dir)
        kb_names.sort()
    except Exception as e:
        logger.warning("Failed to list kb", e)
        kb_names = []
    return kb_names


def refresh_kb_list() -> Dict:
    return gr.update(choices=get_kb_list())


def add_kb(kb_name: str, chatbot: List[List[str]]) -> Tuple[Dict, List[List[str]], Dict]:
    # select_kb, chatbot, kb_delete
    if kb_name is None or kb_name.strip() == "":
        kb_status = "【WARNING】知识库名称不能为空，请重新填写知识库名称"
        chatbot = chatbot + [[None, kb_status]]
        return gr.update(visible=True), \
               chatbot, \
               gr.update(visible=True)
    elif kb_name in get_kb_list():
        kb_status = "【WARNING】与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, kb_status]]
        return gr.update(visible=True), \
               chatbot, \
               gr.update(visible=True)
    else:
        data_dir, pattern = kb_name.rsplit(os.sep, maxsplit=1)
        kb_name = os.path.basename(data_dir)
        kb_path = os.path.join(args.vector_dir, kb_name)
        kb_status = init_embeddings_and_vector_store(kb_path, data_dir, pattern)
        chatbot = chatbot + [[None, kb_status]]
        return gr.update(visible=True, choices=get_kb_list(), value=kb_name), \
               chatbot, \
               gr.update(visible=True, choices=get_kb_list())


def delete_kb(kb_name: str, chatbot: List[List[str]]) -> Tuple[Dict, List[List[str]], Dict]:
    try:
        shutil.rmtree(os.path.join(args.vector_dir, kb_name))
        kb_status = f"成功删除知识库：{kb_name}"
        logger.info(kb_status)
        chatbot = chatbot + [[None, kb_status]]
        return gr.update(choices=get_kb_list()), \
               chatbot, \
               gr.update(choices=get_kb_list())
    except Exception as e:
        kb_status = f"【WARNING】删除知识库：{kb_name}失败"
        logger.error(kb_status, e)
        chatbot = chatbot + [[None, kb_status]]
        return gr.update(visible=True), \
               chatbot, \
               gr.update(visible=True)


def reselect_kb(kb_name: str) -> Dict:
    return gr.update(value=kb_name)


def change_kb(kb_name: str, history: List[List[str]]) -> List[List[str]]:
    kb_path = os.path.join(args.vector_dir, kb_name)
    kb_status = init_embeddings_and_vector_store(kb_path)

    return history + [[None, kb_status]]


def load_summarization_files(files: Union[File, List[File]],
                             history: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    # files, chatbot
    if not isinstance(files, list):
        files = [files]

    loaded_files = [file.name for file in files]
    # global split_documents
    # for file in files:
    # try:
    #     # 导入文本
    #     loader = UnstructuredFileLoader(file.name)
    #     # 将文本转成 Document 对象
    #     document = loader.load()
    #     logger.debug(f'document length: {len(document)}')
    #     # 初始化文本分割器
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=args.chunk_size,
    #         chunk_overlap=args.chunk_overlap
    #     )
    #     # 切分文本
    #     split_documents.extend(text_splitter.split_documents(document))
    #     loaded_files.append(file.name)
    # except Exception as e:
    #     logger.warning(f"Failed to load {file.name}")

    if len(loaded_files):
        file_status = f"已添加如下文件： {'、'.join([os.path.basename(f) for f in loaded_files])} "
    else:
        file_status = "【WARNING】文件上传失败，请重新上传文件"
    logger.info(file_status)

    return loaded_files, \
           history + [[None, file_status]]


def change_task(task: str, history: List[List[str]]) -> Tuple[Dict, Dict, Dict, Dict, List[List[str]]]:
    global args
    args.task = task_zh_to_en[task]
    task_status = initialize_task()
    if task == "知识库问答":
        return gr.update(visible=True), \
               gr.update(visible=True), \
               gr.update(visible=False), \
               gr.update(visible=False), \
               history + [[None, task_status]]
    elif task == "文本摘要":
        return gr.update(visible=False), \
               gr.update(visible=False), \
               gr.update(visible=True), \
               gr.update(visible=False), \
               history + [[None, task_status]]
    elif task == "搜索引擎":
        return gr.update(visible=False), \
               gr.update(visible=False), \
               gr.update(visible=False), \
               gr.update(visible=True), \
               history + [[None, task_status]]
    else:
        return gr.update(visible=False), \
               gr.update(visible=False), \
               gr.update(visible=False), \
               gr.update(visible=False), \
               history + [[None, task_status]]


def init_search(api_key: str, history: List[List[str]]) -> List[List[str]]:
    global args
    args.serp_api_key = api_key
    assert args.task == 'google_search'
    task_status = initialize_task()

    return history + [[None, task_status]]


def get_answer(task: str,
               query: str,
               files: List[str],
               history: List[List[str]]) -> None:
    # logger.info(f"[get_answer] history: {history}")
    if task == "搜索引擎":
        result = langchain_task(prompt=query)
        for resp in [result]:
            reply = "\n\n"
            reply += resp
            history[-1][-1] += reply
            yield history, ""
    elif task == "文本摘要":
        for file in files:
            resp = langchain_task(input_file=file, chunk_size=args.chunk_size,
                                  chunk_overlap=args.chunk_overlap)
            reply = "\n\n"
            reply += resp
            history[-1][-1] += reply
            yield history, ""
    elif task == "知识库问答":
        result = langchain_task(query=query, search_type=args.search_type, k=args.k)
        # logger.info(f"query: {query}, result: {result}")
        for resp in [result]:
            reply = "\n\n"
            source = [
                f"<details>" \
                f"<summary>出处：[{i + 1}] {doc.page_content}</summary>\n" \
                f"{doc.metadata['answer']}\n" \
                f"</details>"
                for i, doc in enumerate(resp["source_documents"])
            ]
            reply += "\n\n".join([f"问：{query}", f"答：{result['result']}"] + source)
            history[-1][-1] += reply
            yield history, ""
    else:
        result = llm(query)
        for resp in [result]:
            reply = "\n\n"
            reply += resp
            history[-1][-1] = reply
            yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME}, task={task}, query={query}, history={history}")
    flag_csv_logger.flag([query, history, task], username=FLAG_USER_NAME)


# 初始化所有模型（LLM, Embeddings, Chain等）
llm_status = initialize_llm()
task_status = initialize_task()

# Gradio配置
with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    task_status = gr.State(task_status)
    llm_status = gr.State(llm_status)
    kb_status = gr.State(default_kb_name)
    file_status = gr.State("")
    gr.Markdown(webui_title)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, llm_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                task = gr.Radio(task_list_zh, label="请选择任务", value=task_en_to_zh[default_task])
                kb_params = gr.Accordion("【知识库问答】参数设定", visible=task_en_to_zh[default_task] == "知识库问答")
                kb_setting = gr.Accordion("【知识库问答】修改知识库", visible=task_en_to_zh[default_task] == "知识库问答")
                summarization_setting = gr.Accordion("【文本摘要】上传文件", visible=task_en_to_zh[default_task] == "文本摘要")
                search_setting = gr.Accordion("【搜索引擎】API KEY", visible=task_en_to_zh[default_task] == "搜索引擎")
                task.change(fn=change_task,
                            inputs=[task, chatbot],
                            outputs=[kb_params, kb_setting, summarization_setting, search_setting, chatbot])
                with kb_params:
                    args.search_threshold = gr.Number(value=args.search_threshold,
                                                      label="召回阈值：相似度超过该值的document才会被召回",
                                                      precision=1,
                                                      interactive=True)
                    args.k = gr.Number(value=args.k, precision=0,
                                       label="召回数量：每次最多召回的document数量", interactive=True)
                    # chunk_conent = gr.Checkbox(value=False,
                    #                            label="是否启用上下文关联",
                    #                            interactive=True)
                    args.chunk_size = gr.Number(value=args.chunk_size, precision=0,
                                                label="最大长度：单段内容的最大长度，超过该值会被切分为不同document",
                                                interactive=True)
                    # chunk_conent.change(fn=change_chunk_conent,
                    #                     inputs=[chunk_conent, gr.Textbox(value="chunk_conent", visible=False), chatbot],
                    #                     outputs=[chunk_sizes, chatbot])
                with kb_setting:
                    kb_select_dropdown = gr.Dropdown(get_kb_list(),
                                                     label="请选择要加载的知识库",
                                                     interactive=True,
                                                     value=get_kb_list()[0] if len(get_kb_list()) > 0 else None)
                    kb_select_button = gr.Button("切换知识库")
                    kb_add_textbox = gr.Textbox(label="请输入新增知识库的原始文件地址（如：path/*.txt）",
                                                lines=1,
                                                interactive=True,
                                                visible=True)
                    kb_add_button = gr.Button(value="新增知识库", visible=True)

                    kb_delete_dropdown = gr.Dropdown(get_kb_list(),
                                                     label="请选择要删除的知识库",
                                                     interactive=True,
                                                     value=get_kb_list()[0] if len(get_kb_list()) > 0 else None)
                    kb_delete_button = gr.Button("删除知识库", visible=True)
                    # select_kb.change(fn=reselect_kb,
                    #                  inputs=select_kb,
                    #                  outputs=select_kb)
                    kb_select_button.click(fn=change_kb,
                                           inputs=[kb_select_dropdown, chatbot],
                                           outputs=chatbot)
                    kb_add_button.click(fn=add_kb,
                                        inputs=[kb_add_textbox, chatbot],
                                        outputs=[kb_select_dropdown, chatbot, kb_delete_dropdown])
                    # delete_kb.change(fn=reselect_kb,
                    #                  inputs=delete_kb,
                    #                  outputs=delete_kb)
                    kb_delete_button.click(fn=delete_kb,
                                           inputs=[kb_delete_dropdown, chatbot],
                                           outputs=[kb_select_dropdown, chatbot, kb_delete_dropdown])
                    flag_csv_logger.setup([task, query, chatbot], "flagged")
                with summarization_setting:
                    file2kb = gr.Column(visible=True)
                    with file2kb:
                        #     # load_kb = gr.Button("加载知识库")
                        #     gr.Markdown("向知识库中添加文件")
                        #     sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                        #                               label="文本入库分句长度限制",
                        #                               interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.doc', '.docx', '.pdf', '.tsv', '.json', ".csv",
                                                        "jsonl"],
                                            file_count="multiple",
                                            show_label=False)
                            load_file_button = gr.Button("上传文件")
                        # with gr.Tab("上传文件夹"):
                        #     folder_files = gr.File(label="添加文件夹",
                        #                            file_count="directory",
                        #                            show_label=False)
                        #     load_folder_button = gr.Button("上传文件夹并加载知识库")
                        # with gr.Tab("删除知识库"):
                        #     files_to_delete = gr.CheckboxGroup(choices=[],
                        #                                        label="删除整个知识库向量文件",
                        #                                        interactive=True)
                        #     delete_file_button = gr.Button("删除整个知识库向量文件")
                    load_file_button.click(load_summarization_files,
                                           show_progress=True,
                                           inputs=[files, chatbot],
                                           outputs=[files, chatbot])
                    # load_folder_button.click(get_vector_store,
                    #                          show_progress=True,
                    #                          inputs=[select_kb, folder_files, sentence_size, chatbot, kb_add,
                    #                                  kb_add],
                    #                          outputs=[kb_path, folder_files, chatbot, files_to_delete], )
                    # delete_file_button.click(delete_file,
                    #                          show_progress=True,
                    #                          inputs=[select_kb, files_to_delete, chatbot],
                    #                          outputs=[files_to_delete, chatbot])
                    flag_csv_logger.setup([task, files, chatbot], "flagged")
                with search_setting:
                    serp_api_key_textbox = gr.Textbox(label="请输入SERP API KEY",
                                                      lines=1,
                                                      interactive=True,
                                                      visible=True)
                    serp_api_key_button = gr.Button(value="确认", visible=True)
                    serp_api_key_button.click(fn=init_search,
                                              inputs=[serp_api_key_textbox, chatbot],
                                              outputs=chatbot)
                    flag_csv_logger.setup([task, query, serp_api_key_textbox, chatbot], "flagged")
                query.submit(get_answer,
                             [task, query, files, chatbot],
                             [chatbot, query])
    # with gr.Tab("知识库测试 Beta"):
    #     with gr.Row():
    #         with gr.Column(scale=10):
    #             chatbot = gr.Chatbot([[None, knowledge_base_test_mode_info]],
    #                                  elem_id="chat-box",
    #                                  show_label=False).style(height=750)
    #             query = gr.Textbox(show_label=False,
    #                                placeholder="请输入提问内容，按回车进行提交").style(container=False)
    #         with gr.Column(scale=5):
    #             mode = gr.Radio(["知识库测试"],  # "知识库问答",
    #                             label="请选择使用模式",
    #                             value="知识库测试",
    #                             visible=False)
    #             knowledge_set = gr.Accordion("知识库设定", visible=True)
    #             kb_setting = gr.Accordion("配置知识库", visible=True)
    #             mode.change(fn=change_mode,
    #                         inputs=[mode, chatbot],
    #                         outputs=[kb_setting, knowledge_set, chatbot])
    #             with knowledge_set:
    #                 score_threshold = gr.Number(value=VECTOR_SEARCH_SCORE_THRESHOLD,
    #                                             label="知识相关度 Score 阈值，分值越低匹配度越高",
    #                                             precision=0,
    #                                             interactive=True)
    #                 vector_search_top_k = gr.Number(value=VECTOR_SEARCH_TOP_K, precision=0,
    #                                                 label="获取知识库内容条数", interactive=True)
    #                 chunk_conent = gr.Checkbox(value=False,
    #                                            label="是否启用上下文关联",
    #                                            interactive=True)
    #                 chunk_sizes = gr.Number(value=CHUNK_SIZE, precision=0,
    #                                         label="匹配单段内容的连接上下文后最大长度",
    #                                         interactive=True, visible=False)
    #                 chunk_conent.change(fn=change_chunk_conent,
    #                                     inputs=[chunk_conent, gr.Textbox(value="chunk_conent", visible=False), chatbot],
    #                                     outputs=[chunk_sizes, chatbot])
    #             with kb_setting:
    #                 kb_refresh = gr.Button("更新已有知识库选项")
    #                 select_kb_test = gr.Dropdown(get_kb_list(),
    #                                              label="请选择要加载的知识库",
    #                                              interactive=True,
    #                                              value=get_kb_list()[0] if len(get_kb_list()) > 0 else None)
    #                 kb_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
    #                                      lines=1,
    #                                      interactive=True,
    #                                      visible=True)
    #                 kb_add = gr.Button(value="添加至知识库选项", visible=True)
    #                 file2kb = gr.Column(visible=False)
    #                 with file2kb:
    #                     # load_kb = gr.Button("加载知识库")
    #                     gr.Markdown("向知识库中添加单条内容或文件")
    #                     sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
    #                                               label="文本入库分句长度限制",
    #                                               interactive=True, visible=True)
    #                     with gr.Tab("上传文件"):
    #                         files = gr.File(label="添加文件",
    #                                         file_types=['.txt', '.md', '.docx', '.pdf'],
    #                                         file_count="multiple",
    #                                         show_label=False
    #                                         )
    #                         load_file_button = gr.Button("上传文件并加载知识库")
    #                     with gr.Tab("上传文件夹"):
    #                         folder_files = gr.File(label="添加文件",
    #                                                # file_types=['.txt', '.md', '.docx', '.pdf'],
    #                                                file_count="directory",
    #                                                show_label=False)
    #                         load_folder_button = gr.Button("上传文件夹并加载知识库")
    #                     with gr.Tab("添加单条内容"):
    #                         one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题", lines=1)
    #                         one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容", lines=5)
    #                         one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
    #                                                                interactive=True)
    #                         load_conent_button = gr.Button("添加内容并加载知识库")
    #                 # 将上传的文件保存到content文件夹下,并更新下拉框
    #                 kb_refresh.click(fn=refresh_kb_list,
    #                                  inputs=[],
    #                                  outputs=select_kb_test)
    #                 kb_add.click(fn=add_kb_name,
    #                              inputs=[kb_name, chatbot],
    #                              outputs=[select_kb_test, kb_name, kb_add, file2kb, chatbot])
    #                 select_kb_test.change(fn=change_kb_name_input,
    #                                       inputs=[select_kb_test, chatbot],
    #                                       outputs=[kb_name, kb_add, file2kb, kb_path, chatbot])
    #                 load_file_button.click(get_vector_store,
    #                                        show_progress=True,
    #                                        inputs=[select_kb_test, files, sentence_size, chatbot, kb_add, kb_add],
    #                                        outputs=[kb_path, files, chatbot], )
    #                 load_folder_button.click(get_vector_store,
    #                                          show_progress=True,
    #                                          inputs=[select_kb_test, folder_files, sentence_size, chatbot, kb_add,
    #                                                  kb_add],
    #                                          outputs=[kb_path, folder_files, chatbot], )
    #                 load_conent_button.click(get_vector_store,
    #                                          show_progress=True,
    #                                          inputs=[select_kb_test, one_title, sentence_size, chatbot,
    #                                                  one_conent, one_content_segmentation],
    #                                          outputs=[kb_path, files, chatbot], )
    #                 flag_csv_logger.setup([query, kb_path, chatbot, mode], "flagged")
    #                 query.submit(get_answer,
    #                              [query, kb_path, chatbot, mode, score_threshold, vector_search_top_k, chunk_conent,
    #                               chunk_sizes],
    #                              [chatbot, query])
    with gr.Tab("模型配置"):
        llm_model = gr.Radio(llm_model_list,
                             label="LLM 模型",
                             value=default_llm_model,
                             interactive=True)
        embedding_model = gr.Radio(embedding_model_list,
                                   label="Embedding 模型",
                                   value=default_embedding_name,
                                   interactive=True)
        # no_remote_model = gr.Checkbox(shared.LoaderCheckPoint.no_remote_model,
        #                               label="加载本地模型",
        #                               interactive=True)
        # llm_history_len = gr.Slider(0, 10,
        #                             value=LLM_HISTORY_LEN,
        #                             step=1,
        #                             label="LLM 对话轮数",
        #                             interactive=True)
        # use_ptuning_v2 = gr.Checkbox(USE_PTUNING_V2,
        #                              label="使用p-tuning-v2微调过的模型",
        #                              interactive=True)
        args.do_sample = gr.Checkbox(args.do_sample,
                                     label="生成参数：do_sample",
                                     interactive=True)
        args.top_p = gr.Slider(0.0, 1.0, value=args.top_p, step=0.1,
                               label="生成参数：top_p", interactive=True)
        args.temperature = gr.Slider(0.0, 5.0, value=args.temperature, step=0.1,
                                     label="生成参数：temperature", interactive=True)
        args.repetition_penalty = gr.Slider(0.0, 5.0, value=args.repetition_penalty, step=0.1,
                                            label="生成参数：repetition_penalty", interactive=True)

        load_model_button = gr.Button("重新加载模型")
        load_model_button.click(reinit_model, show_progress=True,
                                inputs=[llm_model, embedding_model, kb_select_dropdown,
                                        # llm_history_len, no_remote_model, use_ptuning_v2, use_lora,
                                        chatbot],
                                outputs=chatbot)

    demo.load(
        fn=refresh_kb_list,
        inputs=None,
        outputs=kb_select_dropdown,
        queue=True,
    )

(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=7860,
         show_api=False,
         share=False,
         inbrowser=False))
