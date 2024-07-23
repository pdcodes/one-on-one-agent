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
from langchain_community.vectorstores import Qdrant
import os
from datetime import datetime

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# project: str,

def write_to_qdrant(user_email: str, update_content: str):
    # Initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    # Initialize OpenAI embeddings
    # embeddings = OpenAIEmbeddings()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Generate embeddings for the update content
    # vector = embedding_model.embed_query(update_content)
    vectorstore = Qdrant(client, collection_name="one-on-ones", embeddings=embedding_model)

    # Create metadata
    metadata = [{
        "user": user_email,
        # "project": project,
        "date": datetime.now().strftime("%Y-%U")
    }]
    
    vectorstore.add_texts([update_content], metadatas=metadata)


    """

    # Create a unique ID for the update
    # update_id = f"{user_name}_{int(datetime.now().timestamp())}"

    # Add the update to Qdrant with metadata
    client.upsert(
        collection_name="one-on-ones",
        points=[
            models.PointStruct(
                vector=vector,
                payload={**metadata, "content": update_content}
            )
        ]
    )
    """
    print("Update saved successfully.")
    return f"Update saved successfully."