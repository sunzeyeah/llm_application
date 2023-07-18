
import os
import re
from typing import List, Union, Any
from langchain import SerpAPIWrapper
from langchain.agents import (
    load_tools,
    initialize_agent,
    LLMSingleActionAgent,
    AgentExecutor,
    AgentOutputParser,
    AgentType,
)
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.tools import Tool
from langchain.prompts import StringPromptTemplate

from src.tasks.base import Task

TOOL_NAME_EN = "Search"
TOOL_DESCRIPTION_EN = "A search engine. Useful for when you need to answer questions about current events. Input should be a search query."
PREFIX_EN = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS_EN = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX_EN = """Begin!

Question: {input}
Thought: {agent_scratchpad}"""

TOOL_NAME_ZH = "搜索引擎"
TOOL_DESCRIPTION_ZH = "搜索引擎有助于回答即时性问题（如：新闻热点），输入为查询语句。"
PREFIX_ZH = """请回答以下问题，你可以使用如下工具："""
FORMAT_INSTRUCTIONS_ZH = """请使用如下格式：

问题：你需要回答的问题
思考过程：你需要考虑怎么做和如何做
行动：你需要从如下列表中选择一项[{tool_names}]，作为下一步行动
行动输入：行动的输入
结果：采取行动后的结果
... (这个“思考过程/行动/行动输入/结果”的流程可以多次重复)
思考过程：我现在知道了最终答案
最终答案：问题的最终答案"""
SUFFIX_ZH = """开始！

问题：{input}
思考过程：{agent_scratchpad}"""


def custom():
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        )
    ]
    # query = "How many people live in Canada as of 2023?"
    # result = search.run(query)
    # logger.info(result)

    template_search = """
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request and explain what actions were used.
    
    ### Instruction:
    Answer the following questions as best you can. Speak like a priate when you give the Final answer. You have access to the following tools:
    
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s
    
    ### Input:
    {input}
    
    ### Response:
    {agent_scratchpad}
    """

    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]

        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)

    # Replace the template variable with the new Alpaca Instruct template
    template = """
    Please follow the steps below to answer the question using the available tools. Repeat the steps as necessary until you find a solution.
    
    ### Instruction:
    Answer the question: {input}
    You have access to the following tools: {tools}
    
    ### Steps:
    1. Think about the question and the best tool to use.
    2. Perform the action using the selected tool.
    3. Observe the results of the action and provide the final answer.
    
    ### Response Format:
    Thought: Your thought process.
    Action: The name of the tool (one word only, from {tool_names}).
    Action Input: The input you provide to the tool.
    Observation: The results obtained from using the tool.
    Final Answer: The answer to the question based on your observation.
    """

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    class CustomOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )

            regex = r"Action: (.*?)\nAction Input: (.*?)\n"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2).strip()

            return AgentAction(tool=action, tool_input=action_input, log=llm_output)

    class RawOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Any:
            return llm_output

    # llm = ChatGLM2LLM()
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    output_parser = CustomOutputParser()
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor(
        agent=agent,
        agent_output_parser=RawOutputParser(),
        tools=[search],
        name_to_tool_map={"search": search},
        time_limit_secs=60,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


class ZHMRKLOutputParser(AgentOutputParser):

    FINAL_ANSWER_ACTION_ZH = "最终答案："

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS_ZH

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = self.FINAL_ANSWER_ACTION_ZH in text
        regex = (
            r"行动\s*\d*\s*[:：][\s]*(.*?)[\s]*行动\s*\d*\s*输入\s*\d*\s*[:：][\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    "Parsing LLM output produced both a final answer "
                    f"and a parse-able action: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(self.FINAL_ANSWER_ACTION_ZH)[-1].strip()}, text
            )

        if not re.search(r"行动\s*\d*\s*[:：][\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation="Invalid Format: Missing '行动：' after '思考过程：'",
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
                r"[\s]*行动\s*\d*\s*输入\s*\d*\s*[:：][\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation="Invalid Format:"
                            " Missing '行动输入：' after '行动：'",
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "mrkl_zh"


class GoogleSearch(Task):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(llm=kwargs.pop("llm"),
                         languange=kwargs.pop("language", "zh"),
                         verbose=kwargs.pop("verbose", True))
        self.serp_api_key = kwargs.pop("serp_api_key", None)
        os.environ["SERPAPI_API_KEY"] = self.serp_api_key
        self.prefix = PREFIX_ZH if self.language == "zh" else PREFIX_EN
        self.suffix = SUFFIX_ZH if self.language == "zh" else PREFIX_EN
        self.format_instructions = FORMAT_INSTRUCTIONS_ZH if self.language == "zh" else FORMAT_INSTRUCTIONS_EN
        self.output_parser = ZHMRKLOutputParser() if self.language == "zh" else MRKLOutputParser()
        self._init_tools(**kwargs)
        self._init_agent(**kwargs)

    def _init_tools(self, **kwargs: Any) -> None:
        """Initialize Tools"""
        tools = kwargs.get("tools", [])
        self.tool_names = tools
        # self.tools = load_tools(tools)
        self.tools = [
            Tool(
                name=TOOL_NAME_ZH if self.language == "zh" else TOOL_NAME_EN,
                description=TOOL_DESCRIPTION_ZH if self.language == "zh" else TOOL_DESCRIPTION_EN,
                func=SerpAPIWrapper(**kwargs).run,
                coroutine=SerpAPIWrapper(**kwargs).arun,
            )
        ]

    def _init_agent(self, agent: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    **kwargs: Any) -> None:
        """Initialize Agent"""
        agent_kwargs = {
            "prefix": self.prefix,
            "suffix": self.suffix,
            "format_instructions": self.format_instructions,
            "output_parser": self.output_parser
        }
        self.agent = initialize_agent(self.tools, self.llm, agent=agent, verbose=self.verbose,
                                      agent_kwargs=agent_kwargs)

    @property
    def _task_name(self) -> str:
        """Return name of task"""
        return "google_search"

    def __call__(self,
                 prompt: str,
                 **kwargs: Any) -> str:
        """Run Agent"""
        return self.agent.run(prompt)
