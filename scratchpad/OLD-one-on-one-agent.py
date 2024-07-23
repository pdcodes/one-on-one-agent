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
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# ---- CHATBOT/AGENT BEHAVIOR ---- # 
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain.memory.buffer import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from prompts import context_message_prompt
from langchain.memory.buffer import ConversationBufferMemory

# ---- ENV VARIABLES ---- # 
# HF_TOKEN = os.environ["HF_TOKEN"]
# HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
# HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
QDRANT_API = os.environ["QDRANT_API_KEY"]
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "One on One Agent"

# ---- LOAD ENV VARIABLES ---- # 
load_dotenv()

# ---- SETTING UP OUR TOOL BELT ---- # 
# Import our tools
from scratchpad.update_checker import UpdateChecker
update_checker = UpdateChecker()

# from write_to_qdrant import WriteToQdrant

tool_belt = [
    update_checker
]
"""
from langgraph.prebuilt import ToolExecutor
tool_executor = ToolExecutor(tool_belt)
"""
# ---- SET UP CHAT LLM ---- # 
# Prototyping first with Open AI
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
# Setting up Open AI
chat_llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

# Setting up memory
conversation_memory = ConversationBufferMemory(memory_key="chat_history", max_len=200, return_messages=True)

# ---- SET UP STATE OBJECT ---- # 
class AgentState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(AgentState)

def chatbot(state: AgentState):
    return {"messages": [chat_llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# ---- START CHAT ---- #
@cl.on_chat_start
async def one_on_one_update_agent():
    model = chat_llm
    prompt = context_message_prompt
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

# ---- HANDLE MESSAGES AND RESPONES ---- # 
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    # Access chat history messages
    chat_history = "\n".join([msg.content for msg in conversation_memory.chat_memory.messages])

    for chunk in runnable.stream(
        {"chat_history": chat_history, "question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    # Check the response using the UpdateChecker tool
    check_response = update_checker(message.content)
    check_msg = cl.Message(content=check_response.content)
    await check_msg.send()