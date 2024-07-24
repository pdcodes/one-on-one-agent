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

# Setup Chat LLM
from prompts import week_by_week_system_template
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage

chat_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.5,
    streaming=True
)

# Wrap the system prompt in a SystemMessage
system_message = SystemMessage(content=week_by_week_system_template)

# Create a function that prepends the system message to the messages in each call
def chat_with_system(messages):
    return chat_llm([system_message] + messages)

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
    user_email: Optional[str]  # Change from user_name to user_email!

def categorize_input(human_input: str) -> Tuple[str, Optional[str]]:
    prompt = f"""
    Analyze the following user input and determine which category it best fits into:
    - week_time: If the input indicates whether it's the beginning or end of the week
    - email: If the input appears to be the user's email
    - project: Information about the current project
    - accomplishments: Recent achievements or milestones related to the specific project
    - blockers: Issues or challenges faced in completing specific tasks for the project
    - risks: Potential risks to the project's completion or timely delivery
    - personal_updates: Personal news unrelated to the project
    - unclear: If the input doesn't clearly fit into any category

    If the category is "name", also extract the email from the input.
    If the category is "week_time", also extract whether it's the beginning or end of the week.

    User input: {human_input}

    Respond in the following format:
    Category: [category]
    Email: [extracted email if category is "email", otherwise "None"]
    Week Time: [beginning/end if category is "week_time", otherwise "None"]
    """
    
    response = chat_llm.invoke(prompt)
    lines = response.content.strip().split('\n')
    category = lines[0].split(': ')[1].lower()
    name = lines[1].split(': ')[1]
    week_time = lines[2].split(': ')[1]
    
    if category == "week_time":
        return category, week_time
    elif category == "email":
        return category, name if name != "None" else None
    else:
        return category, None

def generate_summary(memory: ConversationBufferMemory) -> str:
    prompt = f"""
    Based on the following conversation, generate a concise summary of the team member's update.

    If the update is from the beginning of a week, it should be formatted as a set of bullets and be generated using the following format:
    
    Beginning of Week:
        Current Tasks:
            Project: The project that the user worked on
            Tasks for the week: The specific tasks that the user will be working on
        Goals for the Week:
            The goals for the week that the user has
        Blockers
            Any blockers, issues, or unknowns that the user might experience
        Personal Update
            Any personal updates from the user

    If the update provided by the user is from the end of the week, it should be formatted as a set of bullets and be generated using the following format:

    End of Week:
        Personal Update
            Any personal updates from the user
        Accomplishments:
            Project: The project that the user worked on
            The tasks that the user completed
        Blockers
            Any blockers, issues, or unknowns that the user experienced this week
        Risks
            Any risks or concerns expressed by the user about the project that they are working on and the goals that have been identified for the project
    
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
    category, detected_value = categorize_input(last_human_message)
    
    # Update the update_state based on the LLM categorization
    if category in update_state:
        update_state[category] = True
    
    # Handle email detection
    if category == "email" and detected_value:
        update_state["email"] = True
        state["user_email"] = detected_value
    elif not update_state.get("email") and detected_value:
        # If email was detected in another category of input
        update_state["email"] = True
        state["user_email"] = detected_value
    
    # Handle week_time detection
    if category == "week_time":
        update_state["is_beginning_of_week"] = detected_value.lower() == "beginning"
    
    # Generate AI response and follow-up question
    chat_history = "\n".join([f"{m.type}: {m.content}" for m in memory.chat_memory.messages])
    
    # Define prompts for beginning and end of week
    beginning_of_week_prompt = """
    For the beginning of the week, focus on:
    1. What project(s) the user is currently working on and what specific tasks are related to the project(s)
    2. What would the user would like to get done by the end of the week
    3. Are there any potential blockers or unknowns that that may come up this week
    4. Did anything really cool happen in the last week that the user would like to share or celebrate
    5. Make sure to collect the user's email
    """
    
    end_of_week_prompt = """
    For the end of the week, focus on:
    1. Did the user finish the following tasks that they set out to do
    2. Did the user accomplish what they wanted to get done by the end of the week? If not, what prevented them from accomplishing that?
    3. How do you feel this week went?
    4. Make sure to collect the user's email
    """
    
    week_specific_prompt = beginning_of_week_prompt if update_state["is_beginning_of_week"] else end_of_week_prompt
    
    # Determine missing information
    missing_info = [key for key, value in update_state.items() if not value and key != "is_beginning_of_week"]
    
    prompt = f"""
    Based on the following conversation and update state, generate a response to the user's last message and a direct question to gather missing information. The response should acknowledge the user's input and transition to the next question naturally but explicitly.

    Chat history:
    {chat_history}

    Last human message: {last_human_message}

    Update state:
    {update_state}

    {week_specific_prompt}

    Missing information: {', '.join(missing_info)}

    Focus on gathering one piece of missing information at a time. Be direct but maintain a conversational tone. If all required information has been gathered, provide a concluding message.

    Response and follow-up question:
    """
    
    response = chat_llm.invoke(prompt)
    ai_message = response.content
    
    memory.chat_memory.add_ai_message(ai_message)
    
    state["next_question"] = ai_message
    state["update_state"] = update_state
    state["memory"] = memory
    state["messages"] = state["messages"] + [AIMessage(content=ai_message)]
    
    return state

# Handle loop
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    update_state = state["update_state"]
    required_fields = ["name", "project", "accomplishments", "blockers", "risks", "personal_updates"]
    if all(update_state.get(field, False) for field in required_fields):
        print("Reached end.")
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
        "is_beginning_of_week": False,
        "email": False,
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
    cl.user_session.set("user_email", None)
    
    await cl.Message(
        """Hello!
        I'm here to help you craft an update for your manager.
        To get started, could you please tell me if this is for the beginning of the week or the end of the week? Please also provide your email address.""").send()

@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    memory = cl.user_session.get("memory")
    update_state = cl.user_session.get("update_state")
    category = cl.user_session.get("category")
    user_email = cl.user_session.get("user_email")  # Change from "user_name" to "user_email"

    print(category)
    print(user_email)
    
    result = graph.invoke({
        "messages": [HumanMessage(content=message.content)],
        "memory": memory,
        "update_state": update_state,
        "last_human_message": message.content,
        "next_question": None,
        "category": category,
        "user_email": user_email,  # Change from "user_name" to "user_email"
    })

    print(f"Result: \n", result)
    
    if should_continue(result) == "end":
        print("We hit end.")
        summary = generate_summary(memory)
        await cl.Message(f"Great! We've completed your update. Here's a summary of what we've discussed:\n\n{summary}\n\nWe'll go ahead and save this update for your manager.").send()
        
        # Save to Qdrant
        user_email = result["user_email"]  # Change from "user_name" to "user_email"
        result = write_to_qdrant(user_email, summary)  # Change from user_name to user_email

    elif isinstance(result, dict):
        if result["next_question"]:
            await cl.Message(content=result["next_question"]).send()
        
        cl.user_session.set("memory", result["memory"])
        cl.user_session.set("update_state", result["update_state"])
        cl.user_session.set("category", result.get("category"))
        cl.user_session.set("user_name", result.get("user_name"))
    else:
        await cl.Message("I'm sorry, but I encountered an unexpected error. Could you please try again?").send()