import glob
from typing import Dict
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    pipeline,
    Pipeline
)
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory

from src.llms import ChatGLMTextGenerationPipeline


def chatglm_auto_configure_device_map(num_gpus: int, model_name: str) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_hidden_layers = 28
    layers_per_gpu = (num_hidden_layers+2) // num_gpus
    layer_prefix = 'transformer'

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上

    encode = ""
    if 'chatglm2' in model_name:
        device_map = {
            f"{layer_prefix}.embedding.word_embeddings": 0,
            f"{layer_prefix}.rotary_pos_emb": 0,
            f"{layer_prefix}.output_layer": 0,
            f"{layer_prefix}.encoder.final_layernorm": 0,
            f"base_model.model.output_layer": 0
        }
        encode = ".encoder"
    else:
        device_map = {f'{layer_prefix}.word_embeddings': 0,
                      f'{layer_prefix}.final_layernorm': 0, 'lm_head': 0,
                      f'base_model.model.lm_head': 0, }
    used = 2
    gpu_target = 0
    for i in range(num_hidden_layers):
        if used >= layers_per_gpu + (gpu_target % 2):
            gpu_target += 1
            gpu_target %= num_gpus
            used = 0
        device_map[f'{layer_prefix}{encode}.layers.{i}'] = gpu_target
        used += 1

    return device_map


def llama_and_baichuan_auto_configure_device_map(num_gpus: int, model_name: str) -> Dict[str, int]:
    layer_prefix = 'model'
    # model.embed_tokens 占用1层
    # model.norm 和 lm_head 占用1层
    # model.layers 占用 num_hidden_layers 层
    # 总共num_hidden_layers+2层分配到num_gpus张卡上
    if "7b" in model_name.lower():
        num_hidden_layers = 32
    elif "13b" in model_name.lower():
        num_hidden_layers = 40
    else:
        raise ValueError(f"Only supports baichuan-7B, baichuan-13B, llama-7B and llama-13B, but {model_name} is provided")

    layers_per_gpu = (num_hidden_layers+2) // num_gpus
    device_map = {f'{layer_prefix}.embed_tokens': 0,
                  f'{layer_prefix}.norm': 0,
                  'lm_head': 0,
                  f'base_model.model.lm_head': 0, }
    used = 2
    gpu_target = 0
    for i in range(num_hidden_layers):
        if used >= layers_per_gpu + (gpu_target % 2):
            gpu_target += 1
            gpu_target %= num_gpus
            used = 0
        device_map[f'{layer_prefix}.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_params_8bit_or_4bit(args, model: PreTrainedModel) -> Dict:
    # init bnb config for quantization
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    if bf16:
        bnb_4bit_compute_dtype = torch.bfloat16
    else:
        bnb_4bit_compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=args.bits == 8,
        load_in_4bit=args.bits == 4,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )
    params = {
        "low_cpu_mem_usage": True,
        'quantization_config': bnb_config
    }
    # infer device map
    if args.multi_card:
        max_memory = get_balanced_memory(model, dtype=torch.int8, low_zero=False,
                                         no_split_module_classes=model._no_split_modules)
        params['device_map'] = infer_auto_device_map(
            model,
            dtype=torch.int8,
            no_split_module_classes=model._no_split_modules,
            max_memory=max_memory
        )
    else:
        params['device_map'] = {"": args.local_rank}

    return params


def load_checkpoint(args, model: PreTrainedModel, strict: bool = True) -> None:
    checkpoints = glob.glob(args.checkpoint.replace("star", "*"))
    st = dict()
    for checkpoint in checkpoints:
        st.update(torch.load(checkpoint, map_location="cpu"))
    model.load_state_dict(st, strict=strict)
    del st


def load(args) -> Pipeline:
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # set eos token
    if "chatglm2" in args.model_name.lower():
        eos_token_id = tokenizer.get_command("eop") if args.checkpoint is not None else tokenizer.get_command("<eos>")
    elif "chatglm1_1" in args.model_name.lower():
        eos_token_id = tokenizer.eos_token_id
    elif "chatglm" in args.model_name.lower():
        eos_token_id = tokenizer.eop_token_id
    elif "baichuan" in args.model_name.lower():
        eos_token_id = tokenizer.bos_token_id if args.checkpoint is not None else tokenizer.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id

    # load model
    if "chatglm" in args.model_name.lower():
        model_class = AutoModelForSeq2SeqLM
    else:
        model_class = AutoModelForCausalLM

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    params = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "torch_dtype": dtype,
        "load_in_8bit": hasattr(args, "bits") and args.bits == 8,
        "load_in_4bit": hasattr(args, "bits") and args.bits == 4,
        # "quantization_config": bnb_config,
        "low_cpu_mem_usage": True
    }
    model = model_class.from_pretrained(args.model_name,
                                        **params)

    # load checkpoint if available
    if args.checkpoint is not None:
        load_checkpoint(args, model)

    model.eval()

    # init huggingface pipeline
    if "chatglm" in args.model_name.lower():
        pipe = ChatGLMTextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            # device=device,
            # device_map={"": args.local_rank} if torch.cuda.is_available() else None,
            max_new_tokens=args.max_length_generation,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            history_length=args.history_length,
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # device=device,
            # device_map={"": args.local_rank} if torch.cuda.is_available() else None,
            max_new_tokens=args.max_length_generation,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )

    return pipe
