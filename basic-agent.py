# ---- IMPORTS ---- # 
# ---- BASIC STUFF ---- # 
import os
from dotenv import load_dotenv
import chainlit as cl

# ---- PROCESSING INPUTS AND OUTPUTS ---- # 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import StrOutputParser

# ---- LLM SETUP ---- # 
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# ---- CHATBOT/AGENT BEHAVIOR ---- # 
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langchain_openai import ChatOpenAI

# ---- ENV VARIABLES ---- # 
# HF_TOKEN = os.environ["HF_TOKEN"]
# HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
# HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
# QDRANT_API = os.environ["QDRANT_API_KEY"]
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]

"""
qdrant_client = QdrantClient(
url="https://20970653-0575-4b50-9e09-91ebdef2f6d3.us-east4-0.gcp.cloud.qdrant.io:6333/",
api_key="RXRLSeAc0CstY0_kCNpR1S2Kq-KxCKyE1TXMPNYVLboXsCcCmYsH8Q",
)
"""


# ---- LOAD ENV VARIABLES ---- # 
load_dotenv()

# ---- SET UP STATE OBJECT ---- # 
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# ---- SET UP CHAT LLM ---- # 
# Using OpenAI while HF is down
"""
chat_llm = HuggingFaceEndpoint(
    endpoint_url=HF_LLM_ENDPOINT,
    top_k=10,
    top_p=0.95,
    temperature=0.5,
    repetition_penalty=1.15,
    huggingfacehub_api_token=HF_TOKEN,
)
"""

chat_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def chatbot(state: State):
    return {"messages": [chat_llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)


graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Agent Awesome:", value["messages"][-1].content)