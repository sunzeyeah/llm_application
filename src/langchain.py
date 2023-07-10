
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")
import os
import argparse
import json
import numpy as np
import torch
import collections
from langchain import (
    HuggingFaceHub,
    LLMChain,
    PromptTemplate
)

from src.utils.logger import logger


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_length_generation", type=int, default=1, help="Maximum number of newly generated tokens")
    parser.add_argument("--checkpoint", type=str)

    # eval
    parser.add_argument("--eval_filename", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--submission_filename", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_few_shot", type=int, default=15, help="Maximum number of examples in few-shot evaulation")
    parser.add_argument("--cot", action="store_true", help="Whether to use Chain of Thought in evaluation")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    return args


def main():

    # from langchain.llms import OpenAI
    # openai_api_key = "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"
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    repo_id = "google/flan-t5-xxl"
    template = """Question: {question} Answer: """
    # initialize Hub LLM
    hub_llm = HuggingFaceHub(
        repo_id=repo_id,
        task="text-generation",
        model_kwargs={'temperature': 1e-10}
    )
    prompt = PromptTemplate(template=template, input_variables=['question'])
    # create prompt template > LLM chain
    llm_chain = LLMChain(prompt=prompt, llm=hub_llm)

    question = "Which NFL team won the Super Bowl in the 2010 season?"
    # ask the user question about NFL 2010
    print(llm_chain.run(question))


if __name__ == "__main__":
    main()
