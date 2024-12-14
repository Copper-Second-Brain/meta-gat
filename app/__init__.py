from fastapi import FastAPI, WebSocket
import torch
from torch_geometric.data import Data
import random
import asyncio
from typing import List

app = FastAPI()

# Sample Data (assuming a pre-trained model exists and user embeddings are available)
num_users = 100
embedding_dim = 128
user_embeddings = torch.randn(num_users, embedding_dim)

# Example: Pre-computed recommendations (dummy data)
# Replace this with real model-based recommendations in practice
def get_recommendations(user_id, top_k=3):
    similarities = torch.cosine_similarity(user_embeddings[user_id].unsqueeze(0), user_embeddings)
    recommended_users = similarities.argsort(descending=True)[1:top_k+1].tolist()  # Skip self
    return recommended_users

# WebSocket endpoint for real-time recommendations
@app.websocket("/ws/recommendations")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Simulate getting recommendations for a random user
            user_id = random.randint(0, num_users - 1)
            recommendations = get_recommendations(user_id)

            # Send data to frontend
            await websocket.send_json({"user_id": user_id, "recommendations": recommendations})
            await asyncio.sleep(.5)  # Adjust for update frequency
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        await websocket.close()
