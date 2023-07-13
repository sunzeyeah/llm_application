
import sys

sys.path.insert(0, "/root/autodl-tmp/Code/llm_application")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/llm_application")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/llm_application")
import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
from langchain.llms import (
    OpenAI,
    # HuggingFacePipeline,
    HuggingFaceHub
)
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from src.utils.logger import logger
from src.utils.file_utils import set_seed
from src.llms import CustomAPI, ChatGLMTextGenerationPipeline, HuggingFacePipeline
from src.tasks import (
    GoogleSearch,
    Summarization,
    ChatBot,
)


def get_parser():
    parser = argparse.ArgumentParser()
    # Required Params
    parser.add_argument("--mode", type=str, required=True, help="openai_api, huggingface_api, custom_api or local,"
                                                                "specify how to call the llm model")
    parser.add_argument("--task", type=str, required=True, help="google_search,"
                                                                "specify the task to perform")
    # Optional Params
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--language", type=str, default="zh", help="prompt使用的语言，一般与模型匹配")
    parser.add_argument("--api_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--verbose", action="store_true", help="是否输出中间结果")
    # generation config
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_length_generation", type=int, default=512, help="Maximum number of newly generated tokens")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    # Task: Search
    parser.add_argument("--serp_api_key", type=str, default=None)
    # Task: Summarization
    parser.add_argument("--input_file", type=str, default=None, help="摘要的外部文件地址")
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    # Task: ChatBot
    parser.add_argument("--embedding_name", type=str, default=None, help="openai or path to huggingface embedding"
                                                                         "embedding method to use")
    parser.add_argument("--vector_dir", type=str, default=None, help="本地知识库的向量文件地址")
    parser.add_argument("--data_dir", type=str, default=None, help="本地知识库原始文件地址")
    parser.add_argument("--pattern", type=str, default=None, help="本地知识库的文件名pattern")

    args = parser.parse_args()

    return args


def init_llm(args):
    if args.mode == "openai_api":
        assert args.api_key is not None, "OPENAI_API_KEY required to init openai api llm"
        os.environ["OPENAI_API_KEY"] = args.api_key
        llm = OpenAI(temperature=args.temperature)
    elif args.mode == "huggingface_api":
        assert args.api_key is not None, "HUGGINGFACEHUB_API_TOKEN required to init hugginface api llm"
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = args.api_key
        llm = HuggingFaceHub(
            repo_id=args.model_name,
            task="text-generation",
            model_kwargs={
                "max_new_tokens": args.max_length_generation,
                "do_sample": args.do_sample,
                "num_return_sequences": args.num_return_sequences,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty
            }
        )
    elif args.mode == "custom_api":
        llm = CustomAPI(
            url=args.api_url,
            max_length=args.max_length,
            do_sample=args.do_sample,
            top_p=args.top_p,
        )
    elif args.mode == "local":
        device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        # init bnb config for quantization
        if torch.cuda.is_available():
            bf16 = torch.cuda.get_device_capability()[0] >= 8
            if bf16:
                bnb_4bit_compute_dtype = torch.bfloat16
            else:
                bnb_4bit_compute_dtype = torch.float16
        else:
            bnb_4bit_compute_dtype = None
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=args.bits == 8,
            load_in_4bit=args.bits == 4,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
        )
        # load model and init pipeline
        if "chatglm" in args.model_name:
            # encoder model structure
            if args.bits in [4, 8]:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                              use_cache=False,
                                                              trust_remote_code=True,
                                                              quantization_config=bnb_config,
                                                              device_map={"": args.local_rank})
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, trust_remote_code=True)
                if args.local_rank >= 0:
                    model = model.half()
            # load checkpoint if available
            if args.checkpoint is not None:
                st = torch.load(args.checkpoint, map_location="cpu")
                model.load_state_dict(st)
                del st
            # set eop token
            if "chatglm2" in args.model_name:
                tokenizer.eop_token_id = tokenizer.get_command("eop") if args.checkpoint is not None else tokenizer.get_command("<eos>")
            # init huggingface pipeline
            pipe = ChatGLMTextGenerationPipeline(
                model=model,
                tokenizer=tokenizer,
                device=device,
                # device_map={"": args.local_rank} if torch.cuda.is_available() else None,
                max_new_tokens=args.max_length_generation,
                eos_token_id=tokenizer.eop_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=args.do_sample,
                num_return_sequences=args.num_return_sequences,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            )
        else:
            # decoder model sturcture
            if args.bits in [4, 8]:
                model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                             use_cache=False,
                                                             trust_remote_code=True,
                                                             quantization_config=bnb_config,
                                                             device_map={"": args.local_rank})
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model_name, use_cache=False, trust_remote_code=True)
                if args.local_rank >= 0:
                    model = model.half()
            # load checkpoint if available
            if args.checkpoint is not None:
                st = torch.load(args.checkpoint, map_location="cpu")
                model.load_state_dict(st)
                del st
            # init huggingface pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                device_map={"": args.local_rank} if torch.cuda.is_available() else None,
                max_new_tokens=args.max_length_generation,
                eos_token_id=tokenizer.bos_token_id,
                do_sample=args.do_sample,
                num_return_sequences=args.num_return_sequences,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            )
        # init langchain llm from huggingface pipeline
        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return llm


def init_task(args, llm):
    if args.task == "google_search":
        kwargs = {"serp_api_key": args.serp_api_key, "tools": ['serpapi']}
        task = GoogleSearch(llm=llm, language=args.language, verbose=args.verbose,
                            **kwargs)
    elif args.task == "summarization":
        task = Summarization(llm=llm, language=args.language, verbose=args.verbose)
    elif args.task == "chatbot":
        if args.embedding_name == "openai":
            embeddings = OpenAIEmbeddings()
        else:
            embedding_device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
            embeddings = HuggingFaceEmbeddings(model_name=args.embedding_name,
                                               model_kwargs={'device': embedding_device})
        task = ChatBot(llm=llm, language=args.language, verbose=args.verbose,
                       embeddings=embeddings, vector_dir=args.vector_dir,
                       data_dir=args.data_dir, pattern=args.pattern,
                       chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    return task


def task_input_params(args):
    if args.task == "google_search":
        input_params = {
            "prompt": args.prompt
        }
    elif args.task == "summarization":
        input_params = {
            "input_file": args.input_file,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap
        }
    elif args.task == "chatbot":
        input_params = {
            "query": args.prompt
        }
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    return input_params


def main():
    args = get_parser()
    logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # init llm
    llm = init_llm(args)

    # init task
    task = init_task(args, llm)

    # execute task
    input_params = task_input_params(args)
    logger.info(task(**input_params))


if __name__ == "__main__":
    main()
