import importlib.util
import torch
import copy

from typing import List, Tuple, Any, Optional, Mapping

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from pydantic import Extra
from transformers import LogitsProcessor
from transformers.pipelines.text_generation import TextGenerationPipeline, ReturnType
from transformers.generation.utils import LogitsProcessorList

from src.utils import logger


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    return response


class ChatGLMTextGenerationPipeline(TextGenerationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if task=='text2text-generation', langchain.huggingface_pipeline will return only generated text,
        # if task=='text-generation', it will return full text including both prompt and generated text
        self.task = "text2text-generation"
        self.history_length = kwargs.pop("history_length", None)

    def __call__(self, text_inputs, **kwargs):
        """
        Complete the prompt(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                One or several prompts (or one list of prompts) to complete.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to return the tensors of predictions (as token indices) in the outputs. If set to
                `True`, the decoded text is not returned.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to return the decoded texts in the outputs.
            return_full_text (`bool`, *optional*, defaults to `True`):
                If set to `False` only added text is returned, otherwise the full text is returned. Only meaningful if
                *return_text* is set to True.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            prefix (`str`, *optional*):
                Prefix added to prompt.
            handle_long_generation (`str`, *optional*):
                By default, this pipelines does not handle long generation (ones that exceed in one form or the other
                the model maximum length). There is no perfect way to adress this (more info
                :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227). This provides common
                strategies to work around that problem depending on your use case.

                - `None` : default strategy where nothing in particular happens
                - `"hole"`: Truncates left of input, and leaves a gap wide enough to let generation happen (might
                  truncate a lot of the prompt and not suitable when generation exceed the model capacity)

            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Return:
            A list or a list of list of `dict`: Returns one of the following dictionaries (cannot return a combination
            of both `generated_text` and `generated_token_ids`):

            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
        """
        return super().__call__(text_inputs, **kwargs)

    def preprocess(self, prompt_text, prefix="", handle_long_generation=None, **generate_kwargs):
        # get history
        history = generate_kwargs.get("history", [])
        if self.history_length is not None and self.history_length > 0:
            history = history[-self.history_length:]
        else:
            history = []
        # concat history with current prompt_text
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, prefix + prompt_text)
        inputs = self.tokenizer(prompt, padding=False, return_tensors=self.framework)

        inputs["prompt_text"] = prompt

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = generate_kwargs.get("max_length", self.model.config.max_length) - cur_len
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -keep_length:]

        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        generate_kwargs.pop("history_length", None)
        generate_kwargs.pop("history", None)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        position_ids = model_inputs.get("position_ids", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        generate_kwargs = copy.deepcopy(generate_kwargs)
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                    "generation_config" in generate_kwargs
                    and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                    "generation_config" in generate_kwargs
                    and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # BS x SL
        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        generate_kwargs["logits_processor"] = logits_processor
        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                 position_ids=position_ids, **generate_kwargs)
        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        # elif self.framework == "tf":
        #     generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}

    def postprocess(self, model_outputs, return_type=ReturnType.NEW_TEXT, clean_up_tokenization_spaces=True):
        generated_sequence = model_outputs["generated_sequence"][0]
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]
        generated_sequence = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:
            if return_type == ReturnType.TENSORS:
                record = {"generated_token_ids": sequence}
            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # Decode text
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )

                if return_type == ReturnType.FULL_TEXT:
                    all_text = prompt_text + text[prompt_length:]
                else:
                    all_text = text[prompt_length:]

                record = {"generated_text": process_response(all_text)}
            records.append(record)

        return records


DEFAULT_MODEL_ID = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation", "summarization")


class HuggingFacePipeline(LLM):
    """Wrapper around HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `text-generation`, `text2text-generation` and `summarization` for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain.llms import HuggingFacePipeline
            hf = HuggingFacePipeline.from_model_id(
                model_id="gpt2",
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 10},
            )
    Example passing pipeline in directly:
        .. code-block:: python

            from langchain.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
            )
            hf = HuggingFacePipeline(pipeline=pipe)
    """

    pipeline: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments passed to the model."""
    pipeline_kwargs: Optional[dict] = None
    """Key word arguments passed to the pipeline."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
            cls,
            model_id: str,
            task: str,
            device: int = -1,
            model_kwargs: Optional[dict] = None,
            pipeline_kwargs: Optional[dict] = None,
            **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            if task == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(model_id, **_model_kwargs)
            elif task in ("text2text-generation", "summarization"):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **_model_kwargs)
            else:
                raise ValueError(
                    f"Got invalid task {task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
        except ImportError as e:
            raise ValueError(
                f"Could not load the {task} model due to missing dependencies."
            ) from e

        if importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if device < -1 or (device >= cuda_device_count):
                raise ValueError(
                    f"Got device=={device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
            if device < 0 and cuda_device_count > 0:
                logger.warning(
                    "Device has %d GPUs available. "
                    "Provide device={deviceId} to `from_model_id` to use available"
                    "GPUs for execution. deviceId is -1 (default) for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )
        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }
        _pipeline_kwargs = pipeline_kwargs or {}
        pipeline = hf_pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_kwargs=_model_kwargs,
            **_pipeline_kwargs,
        )
        if pipeline.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        return cls(
            pipeline=pipeline,
            model_id=model_id,
            model_kwargs=_model_kwargs,
            pipeline_kwargs=_pipeline_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if isinstance(self.pipeline, ChatGLMTextGenerationPipeline):
            history = kwargs.pop("history", [])
            response = self.pipeline(prompt, history=history)
            logger.info(f"[HuggingfacePipeline] history: {history}\nprompt: {prompt}\nresponse: {response}")
        else:
            response = self.pipeline(prompt)
            logger.info(f"[HuggingfacePipeline] prompt: {prompt}\nresponse: {response}")
        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            text = response[0]["generated_text"][len(prompt):]
        elif self.pipeline.task == "text2text-generation":
            text = response[0]["generated_text"]
        elif self.pipeline.task == "summarization":
            text = response[0]["summary_text"]
        else:
            raise ValueError(
                f"Got invalid task {self.pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text
