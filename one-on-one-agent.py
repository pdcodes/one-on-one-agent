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
import pprint
import ast

# ---- CUSTOM LOGIC ---- #
from prompts import chat_prompt
from update_checker import UpdateChecker
from write_to_qdrant import write_to_qdrant

load_dotenv()

# ---- ENV VARIABLES ---- # 
# HF_TOKEN = os.environ["HF_TOKEN"]
# HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
# HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
QDRANT_API = os.environ["QDRANT_API_KEY"]
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "One on One Agent"

# Setup OpenAI
chat_llm = ChatOpenAI(model="gpt-4", temperature=0.7, streaming=True)

# Setup UpdateChecker tool
update_checker = UpdateChecker()

# Setup state
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    memory: ConversationBufferMemory
    update_state: dict
    last_human_message: str
    next_question: Optional[str]
    category: Optional[str]
    user_name: Optional[str]

# Define nodes
from prompts import chat_prompt

def categorize_input(human_input: str) -> Tuple[str, Optional[str]]:
    prompt = f"""
    Analyze the following user input and determine which category it best fits into:
    - name: If the input appears to be the user's name
    - project: Information about the current project
    - accomplishments: Recent achievements or milestones related to the specific project
    - blockers: Issues or challenges faced in completing specific tasks for the project
    - risks: Potential risks to the project's completion or timely delivery
    - personal_updates: Personal news unrelated to the project
    - unclear: If the input doesn't clearly fit into any category

    If the category is "name", also extract the name from the input.

    User input: {human_input}

    Respond in the following format:
    Category: [category]
    Name: [extracted name if category is "name", otherwise "None"]
    """
    
    response = chat_llm.invoke(prompt)
    lines = response.content.strip().split('\n')
    category = lines[0].split(': ')[1].lower()
    name = lines[1].split(': ')[1]
    
    return category, name if name != "None" else None

def generate_summary(memory: ConversationBufferMemory) -> str:
    prompt = f"""
    Based on the following conversation, generate a concise summary of the team member's update. 
    Make sure to include the following attributes:
    - The user's email
    - The projects that they worked on
    - Their accomplishments on those projects
    - The blockers that came up on each project
    - The risks that arose on the project that may undermine company goals
    - Any personal updates
    
    Conversation:
    {memory.chat_memory.messages}
    
    Summary:
    """
    
    summary = chat_llm.invoke(prompt)
    return summary.content

def process_input(state: AgentState) -> AgentState:
    messages = state["messages"]
    memory = state["memory"]
    update_state = state["update_state"]
    
    human_input = messages[-1].content if isinstance(messages[-1], HumanMessage) else ""
    state["last_human_message"] = human_input
    memory.chat_memory.add_user_message(human_input)
    
    # We don't generate an AI response here anymore
    # The AI response will be handled in the check_update function
    
    return {
        **state,
        "memory": memory,
        "update_state": update_state,
    }

def check_update(state: AgentState) -> AgentState:
    update_state = state["update_state"]
    last_human_message = state["last_human_message"]
    memory = state["memory"]
    
    # Use LLM to categorize the input
    category, detected_name = categorize_input(last_human_message)
    
    # Update the update_state based on the LLM categorization
    if category in update_state:
        update_state[category] = True
    
    # Handle name detection
    if category == "name" and detected_name:
        update_state["name"] = True
        state["user_name"] = detected_name
    elif not update_state["name"] and detected_name:
        # If name was detected in another category of input
        update_state["name"] = True
        state["user_name"] = detected_name
    
    # Check if all information has been collected
    if all(update_state.values()):
        state["next_question"] = None
        return state
    
    # Determine the next question to ask
    missing_elements = [key for key, value in update_state.items() if not value]
    next_element = missing_elements[0] if missing_elements else None
    
    prompts = {
        "name": "Could you please tell me your name?",
        "project": "What project are you currently working on?",
        "accomplishments": "What recent accomplishments or achievements have you had on this project?",
        "blockers": "Have you experienced any issues or blockers recently on this project?",
        "risks": "Are there any significant risks that might affect the goals of this project?",
        "personal_updates": "Do you have any notable personal updates you'd like to share?"
    }
    
    # Generate AI response
    chat_history = "\n".join([f"{m.type}: {m.content}" for m in memory.chat_memory.messages])
    response = chat_llm.invoke(
        chat_prompt.format_prompt(
            chat_history=chat_history,
            human_input=last_human_message
        ).to_messages()
    )
    
    ai_message = response.content
    
    # Add the next question if there are still missing elements
    if next_element:
        next_question = prompts[next_element]
        ai_message += f"\n\n{next_question}"
    
    memory.chat_memory.add_ai_message(ai_message)
    
    state["next_question"] = ai_message
    state["update_state"] = update_state
    state["memory"] = memory
    state["messages"] = state["messages"] + [AIMessage(content=ai_message)]
    
    return state

# Build graph
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    # print(f"The state is: \n", state)
    if all(state["update_state"].values()):
        return "end"
    return "continue"

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("process_input", process_input)
workflow.add_node("check_update", check_update)

# Define edges
workflow.set_entry_point("process_input")
workflow.add_edge("process_input", "check_update")

workflow.add_conditional_edges(
    "check_update",
    should_continue,
    {
        "continue": END,  # We end here to wait for the next user input
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
    
    await cl.Message(
        """Hello!
        I'm here to help you craft an update for your manager.
        These updates will include the following details:
        - Your email address
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
    
    result = graph.invoke({
        "messages": [HumanMessage(content=message.content)],
        "memory": memory,
        "update_state": update_state,
        "last_human_message": message.content,
        "next_question": None,
        "category": category,
        "user_name": user_name,
    })

    print(f"Result: \n", result)
    
    if result["next_question"] == None:
        summary = generate_summary(memory)
        await cl.Message(f"Great! We've completed your update. Here's a summary of what we've discussed:\n\n{summary}\n\nWe'll go ahead and save this update for your manager.").send()
        
        # Save to Qdrant
        user_name = result["user_name"]
        # project = result["update_state"].get("project", "Unknown Project")
        result = write_to_qdrant(user_name, summary)

    elif isinstance(result, dict):
        if result["next_question"]:
            await cl.Message(content=result["next_question"]).send()
        
        cl.user_session.set("memory", result["memory"])
        cl.user_session.set("update_state", result["update_state"])
        cl.user_session.set("category", result.get("category"))
        cl.user_session.set("user_name", result.get("user_name"))
    else:
        await cl.Message("I'm sorry, but I encountered an unexpected error. Could you please try again?").send()