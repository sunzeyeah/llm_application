import requests
from abc import ABC
from typing import List, Optional, Any, Mapping

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class CustomAPI(LLM, ABC):

    """API url"""
    url: str = None
    """Key word arguments to pass to the model."""
    model_kwargs: Optional[dict] = None

    @property
    def _llm_type(self) -> str:
        return "custom_api"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any,) -> str:
        # if isinstance(stop, list):
        #     stop = stop + ["\n###", "\nObservation:"]

        _model_kwargs = self.model_kwargs or {}
        data = {**_model_kwargs, **kwargs, "prompt": prompt}
        # do_sample = kwargs.get("do_sample", False)
        # top_p = kwargs.get("top_p", 0.9)
        # temperature = kwargs.get("temperature", 0.0)
        # repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        # max_new_tokens = kwargs.get("max_new_tokens", 512)

        response = requests.post(
            self.url,
            headers={"Content-Type": "application/json"},
            json=data,
        )
        response.raise_for_status()
        return response.json()['response']#[0]['text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"url": self.url},
            **{"model_kwargs": _model_kwargs},
        }
