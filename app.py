# ---- IMPORTS ---- # 
import os
import chainlit as cl
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from dotenv import load_dotenv

import chainlit as cl

# ---- ENV VARIABLES ---- # 
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
# HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]
# QDRANT_API = os.environ["QDRANT_API_KEY"]


# ---- LOAD ENV VARIABLES ---- # 
load_dotenv()

# ---- SET UP CHAT LLM ---- # 
chat_llm = HuggingFaceEndpoint(
    endpoint_url=HF_LLM_ENDPOINT,
    top_k=10,
    top_p=0.95,
    temperature=0.5,
    repetition_penalty=1.15,
    huggingfacehub_api_token=HF_TOKEN,
)

# ---- SET UP EMBEDDINGS / QDRANT ---- # 
"""
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://4d7645f8-ea2b-40be-b446-27506ccf0cf7.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="WixlUGIvJfjbiG-7pvoGNMc_zdLbaVlyvly0MQjQ2CyWS2PU7VWO4Q",
)

print(qdrant_client.get_collections())

url = "<---qdrant cloud cluster url here --->"
api_key = "<---api key here--->"
qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="my_documents",
)

"""

# ---- DEFINE PROMPT TEMPLATE ---- # 
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
            """
            You should play the role of a helpful team member who is eager to hear about what their other team members are working on.
            Your tone throughout the conversation should remain interested and positive.

            The goal of your conversation is to collect information from the user about their projects.

            You should continue to converse with the user until you have the following information from them:
            - Information about what they have recently accomplished on their project
            - Information about things that they are struggling with 
            - Information that would indicate if a project is falling behind

            When you have collected this information, you should recite it back to the user in the following format:
            Section #1: Recent wins and achievements
            This section will include information about what the team member has recently accomplished on their project

            Section #2: Blockers and challenges
            This section will include information about where the team member is struggling and what issues have come up on specific projects

            Section #3: Risks to company goals
            This section will include information about projects that are not moving forward

            Once you have provided the information to the user in this format, you should ask them to confirm if this sounds correct. If it is correct, then you should end the conversation.
            """),
        ("user", "{question}\n"),
    ]
)

# ---- START CHAINLIT ---- # 
@cl.author_rename
def rename(original_author: str):
    rename_dict = {
        "Assistant" : "Updates"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def one_on_one_update_agent():
    model = chat_llm
    prompt = prompt_template
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

# ---- HANDLE MESSAGES AND RESPONES ---- # 
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()