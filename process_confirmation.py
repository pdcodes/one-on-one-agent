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
        return "end"
    
    elif "no" in last_human_message:
        state["messages"].append(AIMessage(content="Alright, I won't save the update. Is there anything else I can help you with?"))
        return "end"
    
    else:
        state["messages"].append(AIMessage(content="I'm sorry, I didn't understand. Please answer with 'Yes' or 'No'. Would you like to save this update?"))
        state["confirmation_state"] = "pending"
        return "continue"
