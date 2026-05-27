from typing import Annotated
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage

load_dotenv()

llm = init_chat_model("groq:qwen/qwen3-32b")

llm

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


def make_tool_graph():
    @tool
    def add(num1: float, num2: float) -> float:
        """Add two numbers."""
        return num1+num2

    tool_node=ToolNode([add])

    llm_with_tools=llm.bind_tools([add])

    def call_llm_model(state: State):
        return {
            "messages": [llm_with_tools.invoke(state["messages"])]
        }

    graph_builder=StateGraph(State)
    graph_builder.add_node("agent", call_llm_model)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition
    )

    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("tools", "agent")

    graph=graph_builder.compile()
    return graph


tool_agent=make_tool_graph()