# CHATBOT WITH TOOL

import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
load_dotenv()


llm = init_chat_model("groq:llama-3.1-8b-instant")

tool=TavilySearch(max_results=3)
# result=tool.invoke("What is langgraph?")
# print(result)

#custom functions

def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: first int
        b: second int
    """
    return a * b

tools=[tool, multiply]

llm_with_tools=llm.bind_tools(tools)

llm_with_tools

### StateGraph

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

memory=MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]


# Node definition
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Graph

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

# Add edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)
builder.add_edge("tools", "tool_calling_llm")

#compile the graph
graph=builder.compile(checkpointer=memory)
config={"configurable": {"thread_id": "chat1"}}
response=graph.invoke({"messages": "Hi, my name is Abdul Rahman Moin."}, config)
response=graph.invoke({"messages": "What is my name?"}, config)

# print("[RESPONSE]: ", response)
# print("response[messages][-1].content: ", response["messages"][-1].content)

# for m in response["messages"]:
    # print(m.pretty_print())



## ====================================== STREAMING ====================================== 

def superbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph=StateGraph(State)

graph.add_node("superbot", superbot)

graph.add_edge(START, "superbot")
graph.add_edge("superbot", END)

graph_builder=graph.compile(checkpointer=memory)


# Invocation

config = {"configurable": {"thread_id": "chat1"}}

graph_builder.invoke({"messages": "Hi, My name is Abdul Rahman Moin"}, config)


config = {"configurable": {"thread_id": "3"}}

for chunk in graph_builder.stream({"messages": "Hi, My name is Abdul Rahman Moin and I Like Cricket."}, config, stream_mode="updates"):
    # print(chunk)
    pass




config = {"configurable": {"thread_id": "4"}}

import asyncio

async def stream_events():
    async for event in graph_builder.astream_events(
        {"messages": "Hi, My name is Abdul Rahman Moin and I Like Cricket."},
        config,
        version="v2",
    ):
        print(event)

asyncio.run(stream_events())