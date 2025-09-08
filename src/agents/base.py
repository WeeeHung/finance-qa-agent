import abc
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from src.agents.types import State


class AgentBase(abc.ABC):
    def __init__(self, memory: InMemorySaver, llm: str, response_format, memory_thread: int, **kwargs) -> None:
        self.memory = memory
        self.config = kwargs
        self.memory_thread = memory_thread
        self.agent = create_react_agent(
            model=llm,
            response_format=response_format,
            tools=[],  # No tools needed for agents
            checkpointer=self.memory,
        )
        self.llm = llm

    def _init_agent(self, resp_format):
        return create_react_agent(
            model=self.llm,
            response_format=resp_format,
            tools=[],  # No tools needed for agents
            checkpointer=self.memory,
        )


    def _call_agent(self, prompt: str) -> dict:
        return self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            {"configurable": {"thread_id": self.memory_thread}}
        )


    @abc.abstractmethod
    def run(self, state: State, feedback: str = '') -> State:
        pass
