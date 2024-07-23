"""
To Do:
- Take the update content, project name, and user as inputs
- Get the current time
- Chunk and embed the update content
- Write to Qdrant with appropriate metadata
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.embeddings import OpenAIEmbeddings
import os
from datetime import datetime

def write_to_qdrant(user_name: str, project: str, update_content: str):
    # Initialize Qdrant client
    client = QdrantClient(
        url=os.environ("QDRANT_URL"),
        api_key=os.environ("QDRANT_API_KEY")
    )

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Generate embeddings for the update content
    vector = embeddings.embed_query(update_content)

    # Create metadata
    metadata = {
        "name": user_name,
        "project": project,
        "date": datetime.now().isoformat()
    }

    # Create a unique ID for the update
    update_id = f"{user_name}_{project}_{int(datetime.now().timestamp())}"

    # Add the update to Qdrant with metadata
    client.upsert(
        collection_name="team_updates",
        points=[
            models.PointStruct(
                id=update_id,
                vector=vector,
                payload={**metadata, "content": update_content}
            )
        ]
    )

    return f"Update saved successfully with ID: {update_id}"