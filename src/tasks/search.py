
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
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.prompts import StringPromptTemplate

from src.tasks.base import Task


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


class GoogleSearch(Task):

    def __init__(self, **kwargs: Any) -> None:
        self.serp_api_key = kwargs.get("serp_api_key", None)
        os.environ["SERPAPI_API_KEY"] = self.serp_api_key
        super().__init__(**kwargs)

    def _init_tools(self, **kwargs: Any) -> None:
        """Initialize Tools"""
        tools = kwargs.get("tools", [])
        self.tool_names = tools
        self.tools = load_tools(tools)

    def _init_agent(self, agent: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose: bool = True,
                    **kwargs: Any) -> None:
        """Initialize Agent"""
        self.agent = initialize_agent(self.tools, self.llm, agent=agent, verbose=verbose)

    @property
    def _task_name(self) -> str:
        """Return name of task"""
        return "google_search"

    def __call__(self, prompt) -> str:
        """Run Agent"""
        return self.agent.run(prompt)
