from abc import abstractmethod
from typing import Any
from langchain.llms.base import LLM


class Task:
    def __init__(self,
                 llm: LLM,
                 **kwargs: Any,
                 # tools: List[str],
                 # agent: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                 # verbose: bool = True
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

    def __call__(self, prompt, **kwargs: Any):
        """Run Agent"""
        self.agent.run(prompt)
