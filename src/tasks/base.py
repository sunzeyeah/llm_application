from abc import abstractmethod
from typing import Any
from langchain.llms.base import LLM


class Task:
    def __init__(self,
                 llm: LLM,
                 language: str = "zh",
                 verbose: bool = True,
                 **kwargs: Any
                 ) -> None:
        self.llm = llm
        self.language = language
        self.verbose = verbose

    @property
    @abstractmethod
    def _task_name(self) -> str:
        """Return name of task"""

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """execute Task"""
