import argparse
import sys

sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/llm_application")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/llm_application")
sys.path.insert(0, "/Users/zeyesun/Documents/Code/llm_application")
sys.path.insert(0, "D:\\Code\\llm_application")
import os
import shutil
import gradio as gr
import uuid
import gc
import torch
import re
from typing import List, Dict, Tuple, Union
from tempfile import NamedTemporaryFile
from gradio.inputs import File
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, VectorStore
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter

from src.utils import logger, rmdir, list_dir, print_gpu_utilization
from src.tasks.chatbot import FAQLoader
from src.tasks import Task
from src.apps import init_llm, init_task


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
    "问答机器人",
    "闲聊",
]
task_en_to_zh = {
    "google_search": "搜索引擎",
    "summarization": "文本摘要",
    "chatbot": "问答机器人",
    "chitchat": "闲聊",
}
task_zh_to_en = {
    "搜索引擎": "google_search",
    "文本摘要": "summarization",
    "问答机器人": "chatbot",
    "闲聊": "chitchat",
}

# Global Variables
llm: LLM = None
langchain_task: Task = None
embeddings: Embeddings = None
vector_store: VectorStore = None


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="local", help="openai_api, huggingface_api, custom_api or local,"
                                                                  "specify how to call the llm model")
    parser.add_argument("--task", type=str, default="chatbot", help="google_search, summarization, chatbot, chitchat"
                                                                    "specify the task to perform")
    parser.add_argument("--model_path", type=str,
                        default=f"D:\\Data\\models" if sys.platform == "win32" else \
                            f"/Users/zeyesun/Documents/Data/models" if sys.platform == "darwin" else \
                            f"/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint"
                            # f"/mnt/sfevol775196/sunzeye273/Data/models"
                        )
    parser.add_argument("--model_name", type=str, default="chatglm2-6B")
    parser.add_argument("--language", type=str, default="zh", help="prompt使用的语言，一般与模型匹配")
    parser.add_argument("--verbose", action="store_true", help="是否输出中间结果")
    parser.add_argument("--device_map", type=str, default="auto", help="device map to allocate model,"
                                                                     "[cpu] means cpu"
                                                                     "[0, 1, 2, ...], number means single-card"
                                                                     "[auto, balanced, balanced_low_0] means multi-card")
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default=None)

    # generation config
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_length_generation", type=int, default=256, help="Maximum number of newly generated tokens")
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--history_length", type=int, default=0, help="Maximum round of history to append to prompt")
    # Task: Summarization
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    # Task: Search
    parser.add_argument("--serp_api_key", type=str, default=None)
    # Task: ChatBot
    parser.add_argument("--embedding_name", type=str, default="text2vec-large-chinese",
                        help="openai or path to huggingface embedding, specify which embedding method to use")
    parser.add_argument("--vector_dir", type=str,
                        default="D:\\Data\\chatgpt\\output\\embeddings" if sys.platform == "win32" else \
                            "/Users/zeyesun/Documents/Data/chatgpt/output/embeddings" if sys.platform == "darwin" else \
                            "/mnt/pa002-28359-vol543625-private/Data/chatgpt/output/embeddings",
                            # "/mnt/sfevol775196/sunzeye273/Data/chatgpt/output/embeddings",
                        help="本地知识库的向量文件根目录")
    parser.add_argument("--kb_name", type=str, default="faq", help="知识库名称")
    parser.add_argument("--k", type=int, default=3, help="number of docs to recall for answering")
    parser.add_argument("--search_type", type=str, default="similarity", help="similarity, similarity_score_threshold, mmr"
                                                                              "metrics used to compare document vectors")
    parser.add_argument("--search_threshold", type=float, default=0.7, help="similarity_score_threshold")
    parser.add_argument("--data_dir", type=str, default=None, help="本地知识库原始文件地址")
    parser.add_argument("--pattern", type=str, default=None, help="本地知识库的文件名pattern")

    args = parser.parse_args()

    return args


def init_embeddings_and_vector_store(vector_dir: str,
                                     data_dir: str = None,
                                     pattern: str = None) -> str:
    global embeddings
    global vector_store
    try:
        # load embedding model
        embedding_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name=os.path.join(args.model_path, args.embedding_name),
                                           model_kwargs={'device': embedding_device})
        # make directory (if vector_dir does not exist)
        if not os.path.exists(vector_dir):
            os.mkdir(vector_dir)
        # init or load vector store
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
        kb_status = f"""【WARNING】知识库：{os.path.basename(vector_dir)}加载失败, {str(e)}"""
        logger.error(kb_status, e)

    return kb_status


def initialize_llm() -> str:
    global llm
    try:
        llm = init_llm(args)
        bits = f"{args.bits}-bit"
        if args.device_map == "cpu":
            loads = "CPU"
        elif args.device_map == "0":
            loads = "单卡"
        elif args.device_map == "auto":
            loads = "多卡（自动负载）"
        elif args.device_map == "balanced":
            loads = "多卡（均衡负载）"
        elif args.device_map == "balanced_low_0":
            loads = "多卡（均衡负载，降低cuda:0）"
        else:
            loads = "多卡（自定义负载）"
        llm_status = f"""LLM模型：{args.model_name}已成功加载，加载模式：{loads} + {bits}"""
    except torch.cuda.OutOfMemoryError as e:
        llm_status = f"""【WARNING】加载LLM模型：{args.model_name} 时发生CUDA out of memory，请开启多卡或者使用8-bit和4-bit"""
        logger.error(llm_status, e)
    except Exception as e:
        llm_status = f"""【WARNING】LLM模型：{args.model_name}加载失败, {str(e)}\n请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.error(llm_status, e)

    return llm_status


def initialize_task() -> str:
    global langchain_task
    try:
        langchain_task = init_task(args, llm, embeddings)
        task_status = f"""任务：{task_en_to_zh[args.task]}已成功加载，可以开始对话"""
    except Exception as e:
        if args.task == "chitchat":
            task_status = f"""任务：{task_en_to_zh[args.task]}已成功加载，可以开始对话"""
        elif args.task == "google_search" and args.serp_api_key is None:
            task_status = f"""【WARNING】任务：{task_en_to_zh[args.task]}默认使用Google，需要SERP_API_KEY，请在右侧输入框内进行输入"""
            logger.warning(task_status)
        else:
            task_status = f"""【WARNING】任务：{task_en_to_zh[args.task]}加载失败, {str(e)}"""
            logger.error(task_status, e)

    return task_status


def update_model_params(
        llm_model: str,
        embedding_model: str,
        kb_name: str,
        device_map: str,
        bits: int,
        checkpoint: str,
        max_length_generation: int,
        do_sample: bool,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        history_length: int,
        history: List[List[str]]) -> List[List[str]]:
    global args
    global llm
    global embeddings
    global vector_store
    args.device_map = device_map
    args.bits = bits
    args.checkpoint = checkpoint
    args.max_length_generation = max_length_generation
    args.do_sample = do_sample
    args.top_p = top_p
    args.temperature = temperature
    args.repetition_penalty = repetition_penalty
    args.history_length = history_length

    # release occupied GPU memory
    if torch.cuda.is_available():
        # print_gpu_utilization("before gpu release", args.local_rank, False)
        try:
            del llm.pipeline.model
            llm.pipeline.model = None
        except AttributeError:
            pass
        try:
            del llm.pipeline.tokenizer
            llm.pipeline.tokenizer = None
        except AttributeError:
            pass
        try:
            del embeddings
            embeddings = None
        except NameError:
            pass
        try:
            del vector_store
            vector_store = None
        except NameError:
            pass
        gc.collect()
        # with torch.cuda.device(f"cuda:{args.local_rank}"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # cuda.select_device(args.local_rank)
        # cuda.close()
        # print_gpu_utilization("after gpu release", args.local_rank, False)

    try:
        args.model_name = llm_model
        args.embedding_name = embedding_model
        # re-init embeddings and vector store
        kb_status = init_embeddings_and_vector_store(vector_dir=os.path.join(args.vector_dir, kb_name))
        logger.debug(kb_status)
        # re-init llm
        llm_status = initialize_llm()
        llm_status = kb_status + "，" + llm_status
        # llm.pipeline._forward_params.update({"do_sample": do_sample, "top_p": top_p,
        #                                      "temperature": temperature, "repetition_penalty": repetition_penalty})
        # llm_status = f"LLM参数已更新，do_sample={do_sample}, top_p={top_p}, temperature={temperature}, " \
        #              f"repetition_penalty={repetition_penalty}"
        logger.debug(llm_status)
    except Exception as e:
        llm_status = f"""【WARNING】LLM模型参数更新失败, {str(e)}"""
        logger.error(llm_status, e)
    return history + [[None, llm_status]]


def get_kb_list() -> List[str]:
    try:
        kb_names = list_dir(args.vector_dir)
        kb_names.sort()
    except Exception as e:
        logger.error("Failed to list kb", e)
        kb_names = []
    return kb_names


def refresh_kb_list() -> Tuple[Dict, Dict]:
    return gr.update(choices=get_kb_list()), gr.update(choices=get_kb_list())


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


def delete_kb(kb_to_delte: str, current_kb: str, chatbot: List[List[str]]) -> Tuple[Dict, List[List[str]], Dict]:
    try:
        assert kb_to_delte != current_kb
        # 删除知识库向量文件地址
        shutil.rmtree(os.path.join(args.vector_dir, kb_to_delte))
        kb_status = f"成功删除知识库：{kb_to_delte}"
        logger.info(kb_status)
        # # 重新加载向量文件 (if necessary)
        # kb_list = get_kb_list()
        # if len(kb_list) > 0:
        #     kb_first = kb_list[0]
        #     kb_path = os.path.join(args.vector_dir, kb_first)
        #     kb_status += "\n" + init_embeddings_and_vector_store(kb_path)
        chatbot = chatbot + [[None, kb_status]]
        return gr.update(choices=get_kb_list(), value=current_kb), \
               chatbot, \
               gr.update(choices=get_kb_list(), value=current_kb)
    except AssertionError as e:
        kb_status = f"【WARNING】待删除知识库与当前使用中知识库相同，无法删除，请换成其他知识库"
        logger.error(kb_status, e)
        chatbot = chatbot + [[None, kb_status]]
        return gr.update(visible=True), \
               chatbot, \
               gr.update(visible=True)
    except Exception as e:
        kb_status = f"【WARNING】删除知识库：{kb_to_delte}失败, {str(e)}"
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
    if task == "问答机器人":
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


def update_kb_params(score: float,
                     k: int,
                     chunk_size: int,
                     history: List[List[str]]) -> List[List[str]]:
    global args
    args.search_threshold = score
    args.k = k
    args.chunk_size = chunk_size
    status = f"""知识库参数已更新，召回阈值={score}, 召回数量={k}, 单段内容的最大长度(chunk_size)={chunk_size}"""
    return history + [[None, status]]


@torch.no_grad()
def get_answer(task: str,
               history: List[List[str]],
               query: str = None,
               files: List[NamedTemporaryFile] = None) -> None:
    try:
        if task == "搜索引擎":
            result = langchain_task(prompt=query)
            for resp in [result]:
                reply = f"问：{query}\n\n答：{resp}"
                history.append([None, reply])
                yield history, ""
        elif task == "文本摘要":
            for file in files:
                resp = langchain_task(input_file=file.name, chunk_size=args.chunk_size,
                                      chunk_overlap=args.chunk_overlap)
                reply = f"摘要：{resp}"
                history.append([None, reply])
                yield history, ""
        elif task == "问答机器人":
            result = langchain_task(query=query, search_type=args.search_type, k=args.k)
            for resp in [result]:
                source = [
                    f"<details>" \
                    f"<summary>出处：[{i + 1}] {doc.page_content}</summary>\n" \
                    f"{doc.metadata['answer']}\n" \
                    f"</details>"
                    for i, doc in enumerate(resp["source_documents"])
                ]
                reply = "\n\n".join([f"问：{query}", f"答：{result['result']}"] + source)
                history.append([None, reply])
                yield history, ""
        else:
            # only select the most recent chitchat history, search or chatbot history is not selected
            pattern = r"(.*?)问：[\s]*(.*?)答：[\s]*(.*)"
            for idx in range(len(history)-1, -1, -1):
                _, text = history[idx]
                if "闲聊已成功加载" in text:
                    break
            else:
                idx = len(history)
            dialog_history = []
            for i in range(idx, len(history)):
                _, text = history[i]
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    old_query = match.group(2).strip()
                    old_reponse = match.group(3).strip()
                    dialog_history.append((old_query, old_reponse))
            logger.debug(f"dialog_history: {dialog_history}")
            result = llm(query, history=dialog_history)
            for resp in [result]:
                reply = f"问：{query}\n\n答：{resp}"
                history[-1][-1] += "\n\n" + reply
                yield history, ""
    except torch.cuda.OutOfMemoryError as e:
        reply = f"""【WARNING】计算答案时发生CUDA out of memory，请开启多卡或者使用8-bit和4-bit"""
        history.append([None, reply])
        yield history, ""
    except Exception as e:
        logger.error("获取答案失败", e)
        reply = str(e)
        history.append([None, reply])
        yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME}, task={task}, query={query}, history={history}")
    flag_csv_logger.flag([task, query, history], username=FLAG_USER_NAME)


# LangChain and LLM Params
args = get_parser()
init_message = f"""欢迎使用 LLM Application Web UI！

请在右侧切换任务，目前支持{len(task_list_zh)}类：{" ".join([f"({i + 1}) {t}" for i, t in enumerate(task_list_zh)])}

当前任务：{task_en_to_zh[args.task]}
当前LLM模型：{args.model_name}
当前embedding模型：{args.embedding_name}
当前知识库：{args.kb_name}
"""
llm_model_list = [
    "chatglm2-6B",
    "chatglm2-6B-int4",
    "chatglm-6B",
    "vicuna-7B-v1.1",
    "baichuan-13B-chat",
    "bloomz-560M",
    "llama-7B",
    "llama-13B",
    "llama2-7B-chat",
    "llama2-13B-chat",
]
embedding_model_list = [
    "text2vec-large-chinese",
]

# 初始化所有模型（LLM, Embeddings, Chain等）
llm_status = initialize_llm()
task_status = initialize_task()

# Gradio配置
with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    task_status = gr.State(task_status)
    llm_status = gr.State(llm_status)
    kb_status = gr.State(args.kb_name)
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
                task = gr.Radio(task_list_zh, label="请选择任务", value=task_en_to_zh[args.task])
                kb_params = gr.Accordion("【问答机器人】参数设定", visible=task_en_to_zh[args.task] == "问答机器人")
                kb_setting = gr.Accordion("【问答机器人】修改知识库", visible=task_en_to_zh[args.task] == "问答机器人")
                summarization_setting = gr.Accordion("【文本摘要】上传文件", visible=task_en_to_zh[args.task] == "文本摘要")
                search_setting = gr.Accordion("【搜索引擎】API KEY", visible=task_en_to_zh[args.task] == "搜索引擎")
                task.change(fn=change_task,
                            inputs=[task, chatbot],
                            outputs=[kb_params, kb_setting, summarization_setting, search_setting, chatbot])
                with kb_params:
                    search_threshold_slider = gr.Slider(0.0, 1.0, value=args.search_threshold, step=0.1,
                                                        label="召回阈值：相似度超过该值的document才会被召回", interactive=True)
                    # search_threshold_number = gr.Number(value=args.search_threshold, minimum=0.0, maximum=1.0, precision=1,
                    #                                     label="召回阈值：相似度超过该值的document才会被召回", interactive=True)
                    # k_number = gr.Number(value=args.k, precision=0,
                    #                      label="召回数量：每次最多召回的document数量", interactive=True)
                    k_slider = gr.Slider(1, 10, value=args.k, step=1,
                                         label="召回数量：每次最多召回的document数量", interactive=True)
                    # chunk_conent = gr.Checkbox(value=False,
                    #                            label="是否启用上下文关联",
                    #                            interactive=True)
                    # chunk_size_number = gr.Number(value=args.chunk_size, precision=0, minimum=64, maximum=4096,
                    #                               label="最大长度：单段内容的最大长度，超过该值会被切分为不同document", interactive=True)
                    chunk_size_slider = gr.Slider(64, 4096, value=args.chunk_size, step=1,
                                                  label="最大长度：单段内容的最大长度，超过该值会被切分为不同document", interactive=True)
                    update_kb_params_button = gr.Button("更新知识库参数")
                    update_kb_params_button.click(fn=update_kb_params,
                                                  inputs=[search_threshold_slider, k_slider, chunk_size_slider,
                                                          chatbot],
                                                  outputs=chatbot)
                with kb_setting:
                    kb_select_dropdown = gr.Dropdown(get_kb_list(),
                                                     label="请选择要加载的知识库",
                                                     interactive=True,
                                                     value=args.kb_name)
                    kb_select_button = gr.Button("切换知识库")
                    kb_add_textbox = gr.Textbox(label='请输入新增知识库的原始文件地址',
                                                info='路径格式：path/{kb_name}/*.jsonl，文件中每行的格式：{"prompt": "", "label": ""}）',
                                                lines=1,
                                                interactive=True)
                    kb_add_button = gr.Button(value="新增知识库")
                    kb_delete_dropdown = gr.Dropdown(get_kb_list(),
                                                     label="请选择要删除的知识库",
                                                     interactive=True,
                                                     value=get_kb_list()[0] if len(get_kb_list()) > 0 else None,
                                                     visible=True)
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
                                           inputs=[kb_delete_dropdown, kb_select_dropdown, chatbot],
                                           outputs=[kb_select_dropdown, chatbot, kb_delete_dropdown])
                    flag_csv_logger.setup([task, query, chatbot], "flagged")
                with summarization_setting:
                    file2kb = gr.Column()
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
                    # flag_csv_logger.setup([task, files, chatbot], "flagged")
                with search_setting:
                    serp_api_key_textbox = gr.Textbox(label="请输入SERP API KEY",
                                                      lines=1,
                                                      interactive=True)
                    serp_api_key_button = gr.Button(value="确认")
                    serp_api_key_button.click(fn=init_search,
                                              inputs=[serp_api_key_textbox, chatbot],
                                              outputs=chatbot)
                    # flag_csv_logger.setup([task, query, serp_api_key_textbox, chatbot], "flagged")
                query.submit(get_answer,
                             [task, chatbot, query, files],
                             [chatbot, query])
    with gr.Tab("模型配置"):
        llm_model_dropdown = gr.Dropdown(llm_model_list,
                                         label="LLM 模型",
                                         value=args.model_name,
                                         interactive=True,
                                         visible=True)
        embedding_model_dropdown = gr.Dropdown(embedding_model_list,
                                               label="Embedding 模型",
                                               value=args.embedding_name,
                                               interactive=True,
                                               visible=True)
        bits_radio = gr.Radio([4, 8, 16, 32], value=args.bits,
                              label="模型加载bit数",
                              interactive=True)
        device_map_radio = gr.Radio(["cpu", "0", "auto", "balanced", "balanced_low_0", "custom"],
                                    value=args.device_map,
                                    label="是否使用GPU、是否开启多卡以及多卡负载管理",
                                    info="cpu-CPU，0-单卡，auto-多卡（自动负载），balanced-多卡（均衡负载），balanced_low_0-多卡（均衡负载，降低cuda:0），custom-多卡（自定义负载）",
                                    interactive=True)
        checkpoint_textbox = gr.Textbox(value=args.checkpoint, label="模型checkpoint文件名", interactive=True)
        max_length_generation_slider = gr.Slider(8, 4096, value=args.max_length_generation, step=1,
                                                 label="生成参数：max_new_tokens",
                                                 interactive=True)
        do_sample_checkbox = gr.Checkbox(args.do_sample,
                                         label="生成参数：do_sample",
                                         interactive=True)
        top_p_slider = gr.Slider(0.0, 1.0, value=args.top_p, step=0.01,
                                 label="生成参数：top_p", interactive=True)
        temperature_slider = gr.Slider(0.0, 5.0, value=args.temperature, step=0.1,
                                       label="生成参数：temperature", interactive=True)
        repetition_penalty_slider = gr.Slider(0.0, 5.0, value=args.repetition_penalty, step=0.1,
                                              label="生成参数：repetition_penalty", interactive=True)
        history_length_slider = gr.Slider(0, 10, value=args.history_length, step=1,
                                          label="最大历史轮数：history_length",
                                          info="目前仅用于闲聊任务和ChatGLM类模型", interactive=True)
        update_model_params_button = gr.Button("更新参数并重新加载模型")
        update_model_params_button.click(update_model_params,
                                         show_progress=True,
                                         inputs=[llm_model_dropdown, embedding_model_dropdown, kb_select_dropdown, device_map_radio,
                                                 bits_radio, checkpoint_textbox, max_length_generation_slider, do_sample_checkbox, top_p_slider,
                                                 temperature_slider, repetition_penalty_slider, history_length_slider, chatbot],
                                         outputs=chatbot)

    demo.load(
        fn=refresh_kb_list,
        inputs=None,
        outputs=[kb_select_dropdown, kb_delete_dropdown],
        queue=True,
    )

(demo
 .queue(concurrency_count=8)
 .launch(server_name='0.0.0.0',
         server_port=7860,
         show_api=False,
         share=False,
         inbrowser=False))
