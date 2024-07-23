# ---- IMPORTS ---- # 
import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Tuple, Union, Optional, Literal
from langgraph.graph.message import add_messages

# ---- CUSTOM LOGIC ---- #
from prompts import chat_prompt
# from update_checker import UpdateChecker # Not using this anymore
from write_to_qdrant import write_to_qdrant

load_dotenv()

# ---- ENV VARIABLES ---- # 
# HF_TOKEN = os.environ["HF_TOKEN"]
# HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
# HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
QDRANT_API = os.environ["QDRANT_API_KEY"]
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "One on One Agent w/ Qdrant"

# Setup OpenAI
chat_llm = ChatOpenAI(model="gpt-4", temperature=0.7, streaming=True)

# Setup UpdateChecker tool
# update_checker = UpdateChecker()

# Setup state
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    memory: ConversationBufferMemory
    update_state: dict
    last_human_message: str
    next_question: Optional[str]
    category: Optional[str]
    user_name: Optional[str]
    last_question: Optional[str]
    confirmation_state: Optional[str]

# Define nodes
from prompts import chat_prompt

def categorize_input(human_input: str) -> Tuple[List[str], Optional[str]]:
    prompt = f"""
    Analyze the following user input and determine which categories it fits into:
    - name: If the input appears to contain the user's name
    - project: Information about the current project
    - accomplishments: Recent achievements or milestones
    - blockers: Issues or challenges faced
    - risks: Potential risks to the project
    - personal_updates: Personal news unrelated to the project
    - unclear: If the input doesn't clearly fit into any category

    The input may fit into multiple categories. List all that apply.
    If the category includes "name", also extract the name from the input.

    User input: {human_input}

    Respond in the following format:
    Categories: [comma-separated list of categories]
    Name: [extracted name if "name" is in categories, otherwise "None"]
    """
    
    response = chat_llm.invoke(prompt)
    lines = response.content.strip().split('\n')
    
    categories = []
    name = None
    
    for line in lines:
        if line.startswith("Categories:"):
            categories_part = line.split(":", 1)
            if len(categories_part) > 1:
                categories = [cat.strip().lower() for cat in categories_part[1].split(",")]
        elif line.startswith("Name:"):
            name_part = line.split(":", 1)
            if len(name_part) > 1:
                extracted_name = name_part[1].strip()
                if extracted_name.lower() != "none":
                    name = extracted_name
    
    if not categories:
        categories = ["unclear"]
    
    return categories, name

def process_input(state: AgentState) -> AgentState:
    messages = state["messages"]
    memory = state["memory"]
    update_state = state["update_state"]
    
    human_input = messages[-1].content if isinstance(messages[-1], HumanMessage) else ""
    
    if not human_input.strip():
        return {
            **state,
            "last_human_message": "",
            "next_question": "I'm sorry, but I didn't receive any input. Could you please try again?"
        }
    
    memory.chat_memory.add_user_message(human_input)
    
    return {
        **state,
        "memory": memory,
        "update_state": update_state,
        "last_human_message": human_input,
        "next_question": None  # Reset next_question after processing input
    }

def check_update(state: AgentState) -> AgentState:
    update_state = state["update_state"]
    last_human_message = state["last_human_message"]
    memory = state["memory"]

    categories, detected_name = categorize_input(last_human_message)
    
    for category in categories:
        if category in update_state and category != "unclear":
            update_state[category] = True
    
    if "name" in categories or (not update_state["name"] and detected_name):
        update_state["name"] = True
        state["user_name"] = detected_name
    
    chat_history = "\n".join([f"{m.type}: {m.content}" for m in memory.chat_memory.messages])
    response = chat_llm.invoke(
        chat_prompt.format_prompt(
            chat_history=chat_history,
            human_input=last_human_message
        ).to_messages()
    )
    
    ai_message = response.content
    
    missing_elements = [key for key, value in update_state.items() if not value]
    if missing_elements:
        next_element = missing_elements[0]
        prompts = {
            "name": "Could you please tell me your name?",
            "project": "What project are you currently working on?",
            "accomplishments": "What recent accomplishments or achievements have you had on this project?",
            "blockers": "Have you experienced any issues or blockers recently on this project?",
            "risks": "Are there any significant risks that might affect the goals of this project?",
            "personal_updates": "Do you have any notable personal updates you'd like to share?"
        }
        next_question = prompts[next_element]
        ai_message += f"\n\n{next_question}"
        state["last_question"] = next_question
    else:
        summary = generate_summary(memory)
        ai_message += f"\n\nGreat! Here's a summary of your update:\n\n{summary}\n\nWould you like me to save this update? Please respond with Yes or No."
        state["confirmation_state"] = "pending"
    
    memory.chat_memory.add_ai_message(ai_message)
    
    return {
        **state,
        "update_state": update_state,
        "memory": memory,
        "messages": state["messages"] + [AIMessage(content=ai_message)],
        "category": categories,
    }

def generate_summary(memory: ConversationBufferMemory) -> str:
    prompt = f"""
    Based on the following conversation, generate a concise summary of the team member's update. 
    Include their name, project, accomplishments, blockers, risks, and any personal updates.
    
    Conversation:
    {memory.chat_memory.messages}
    
    Summary:
    """
    
    summary = chat_llm.invoke(prompt)
    return summary.content

def process_confirmation(state: AgentState) -> str:
    last_human_message = state["last_human_message"].lower()
    if "yes" in last_human_message:
        # Generate summary
        summary = generate_summary(state["memory"])
        
        # Save to Qdrant
        user_name = state["user_name"]
        project = state["update_state"].get("project", "Unknown Project")
        result = write_to_qdrant(user_name, project, summary)
        
        state["messages"].append(AIMessage(content=f"Great! {result} Is there anything else I can help you with?"))
        state["confirmation_state"] = None  # Reset confirmation state
        return "continue"  # Changed to "continue" to allow for further interaction
    
    elif "no" in last_human_message:
        state["messages"].append(AIMessage(content="Alright, I won't save the update. Is there anything else I can help you with?"))
        state["confirmation_state"] = None  # Reset confirmation state
        return "continue"  # Changed to "continue" to allow for further interaction
    
    else:
        state["messages"].append(AIMessage(content="I'm sorry, I didn't understand. Please answer with 'Yes' or 'No'. Would you like to save this update?"))
        state["confirmation_state"] = "pending"
        return "continue"

# Manage the cycle
def should_continue(state: AgentState) -> Literal["process_input", "check_update", "confirm", "end"]:
    if state.get("confirmation_state") == "pending":
        return "confirm"
    elif all(state["update_state"].values()):
        return "confirm"
    elif state["last_human_message"].strip() == "":
        return "end"  # Add this condition to break the loop when there's no input
    else:
        return "process_input"  # Changed from "check_update" to "process_input"

# Build graph
# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("process_input", process_input)
workflow.add_node("check_update", check_update)
workflow.add_node("process_confirmation", process_confirmation)

workflow.set_entry_point("process_input")

workflow.add_conditional_edges(
    "process_input",
    should_continue,
    {
        "process_input": "check_update",  # Changed this line
        "confirm": "process_confirmation",
        "end": END,
    }
)

workflow.add_conditional_edges(
    "check_update",
    should_continue,
    {
        "process_input": "process_input",
        "confirm": "process_confirmation",
        "end": END,
    }
)

workflow.add_conditional_edges(
    "process_confirmation",
    lambda x: x,
    {
        "continue": "process_input",
        "end": END,
    }
)

graph = workflow.compile()

# Chainlit handlers
@cl.on_chat_start
async def start():
    update_state = {
        "name": False,
        "project": False,
        "accomplishments": False,
        "blockers": False,
        "risks": False,
        "personal_updates": False
    }
    memory = ConversationBufferMemory(return_messages=True)
    cl.user_session.set("graph", graph)
    cl.user_session.set("memory", memory)
    cl.user_session.set("update_state", update_state)
    cl.user_session.set("category", None)
    cl.user_session.set("user_name", None)
    cl.user_session.set("last_question", None)
    cl.user_session.set("confirmation_state", None)
    
    await cl.Message(
        """Hello! 
        I'm here to help you craft an update for your manager.
        These updates will include the following details:
        - Your name
        - The project that you're working on
        - Any recent accomplishments or wins that you've had on that project
        - Any blockers or issues that you've faced on that project
        - Any notable risks about the project that should be escalated
        - Any personal updates not related to this project
        To get started, could you please tell me your name?""").send()

@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    memory = cl.user_session.get("memory")
    update_state = cl.user_session.get("update_state")
    category = cl.user_session.get("category")
    user_name = cl.user_session.get("user_name")
    confirmation_state = cl.user_session.get("confirmation_state")
    
    result = graph.invoke({
        "messages": [HumanMessage(content=message.content)],
        "memory": memory,
        "update_state": update_state,
        "last_human_message": message.content,
        "category": category,
        "user_name": user_name,
        "confirmation_state": confirmation_state,
    })
    
    if result == END:
        await cl.Message("Thank you for using the update assistant. Is there anything else I can help you with?").send()
    elif isinstance(result, dict):
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                await cl.Message(content=last_message.content).send()
        
        cl.user_session.set("memory", result.get("memory", memory))
        cl.user_session.set("update_state", result.get("update_state", update_state))
        cl.user_session.set("category", result.get("category", category))
        cl.user_session.set("user_name", result.get("user_name", user_name))
        cl.user_session.set("confirmation_state", result.get("confirmation_state", confirmation_state))
    else:
        await cl.Message("I'm sorry, but I encountered an unexpected error. Could you please try again?").send()