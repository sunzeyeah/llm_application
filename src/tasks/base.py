from abc import abstractmethod
from typing import Any
from langchain.llms.base import LLM


class Task:
    def __init__(self,
                 llm: LLM,
                 **kwargs: Any
                 ) -> None:
        self.llm = llm
        self._init_tools(**kwargs)
        self._init_agent(**kwargs)

    @property
    @abstractmethod
    def _task_name(self) -> str:
        """Return name of task"""

    @abstractmethod
    def _init_tools(self, **kwargs: Any) -> None:
        """Initialize Tools"""

    @abstractmethod
    def _init_agent(self, **kwargs: Any) -> None:
        """Initialize Agent"""

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """execute Task"""
