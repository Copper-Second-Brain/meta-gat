# backend/main.py

from fastapi import FastAPI, WebSocket
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import pandas as pd
import random
import asyncio
from typing import List, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np

app = FastAPI()

# Define the GNN Model (Must match the training model)
class PatientGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(PatientGNN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.attention = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.gcn2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.attention(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        return x

# Function to load and preprocess data
def load_preprocess_data(
    filepath: str,
    sample_size: int = 50,
    k: int = 10,
    random_state: int = 42,
    encoders: Dict[str, LabelEncoder] = None,
    scaler: StandardScaler = None
):
    """
    Load the dataset, sample a subset, encode categorical features,
    normalize continuous features, and create an optimized edge list.

    Args:
        filepath (str): Path to the CSV dataset.
        sample_size (int): Number of samples to select.
        k (int): Number of neighbors to connect for each node.
        random_state (int): Seed for reproducibility.
        encoders (Dict[str, LabelEncoder], optional): Pre-fitted LabelEncoders.
        scaler (StandardScaler, optional): Pre-fitted StandardScaler.

    Returns:
        Data: PyTorch Geometric Data object with sampled data.
        Dict[str, LabelEncoder]: Encoders used for categorical features.
        StandardScaler: Scaler used for continuous features.
        pd.DataFrame: Sampled DataFrame.
    """
    print("Loading dataset...")
    # Load dataset
    df = pd.read_csv(filepath)
    print(f"Original dataset size: {df.shape}")

    # Sample the dataset
    print("Sampling data...")
    if sample_size < len(df):
        df_sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        df_sampled = df.reset_index(drop=True)
    print(f"Sampled dataset size: {df_sampled.shape}")

    # Encode categorical features
    print("Encoding categorical features...")
    if encoders is None:
        encoders = {}
        categorical_cols = ['disease_family', 'disease', 'smoking_status', 'cough_type']
        for col in categorical_cols:
            if col in df_sampled.columns:
                le = LabelEncoder()
                df_sampled[col] = le.fit_transform(df_sampled[col])
                encoders[col] = le
                print(f"Encoded column: {col}")
            else:
                raise ValueError(f"Column '{col}' not found in the dataset.")
    else:
        # Use pre-fitted encoders
        categorical_cols = ['disease_family', 'disease', 'smoking_status', 'cough_type']
        for col in categorical_cols:
            if col in df_sampled.columns:
                if col in encoders:
                    df_sampled[col] = encoders[col].transform(df_sampled[col])
                    print(f"Transformed column with pre-fitted encoder: {col}")
                else:
                    raise ValueError(f"Encoder for column '{col}' not provided.")
            else:
                raise ValueError(f"Column '{col}' not found in the dataset.")

    # Normalize continuous features
    print("Normalizing continuous features...")
    continuous_cols = ['systolic_bp', 'diastolic_bp', 'cholesterol', 'BMI', 'heart_rate', 'blood_glucose']
    for col in continuous_cols:
        if col not in df_sampled.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
    
    if scaler is None:
        scaler = StandardScaler()
        df_sampled[continuous_cols] = scaler.fit_transform(df_sampled[continuous_cols])
        print(f"Normalized columns: {continuous_cols}")
    else:
        # Use pre-fitted scaler
        df_sampled[continuous_cols] = scaler.transform(df_sampled[continuous_cols])
        print(f"Transformed columns with pre-fitted scaler: {continuous_cols}")

    # Prepare node features (X)
    print("Preparing node features...")
    if 'patient_id' in df_sampled.columns:
        feature_cols = [col for col in df_sampled.columns if col != 'patient_id']
        x = torch.tensor(df_sampled[feature_cols].values, dtype=torch.float32)
    else:
        x = torch.tensor(df_sampled.values, dtype=torch.float32)
    print(f"Node features shape: {x.shape}")

    # Create optimized edge list
    print("Creating edge list...")
    edge_index = create_edge_list_vectorized(df_sampled, k=k)
    print(f"Edge list created with shape: {edge_index.shape}")

    # Create Data object
    data = Data(x=x, edge_index=edge_index)

    return data, encoders, scaler, df_sampled

def create_edge_list_vectorized(df: pd.DataFrame, k: int = 10):
    """
    Optimized edge list creation using vectorized operations.
    Connect each node to k randomly selected neighbors within the same 'disease_family'.

    Args:
        df (pd.DataFrame): Sampled DataFrame.
        k (int): Number of neighbors to connect for each node.

    Returns:
        torch.Tensor: Edge index tensor of shape [2, num_edges].
    """
    print("Grouping data by 'disease_family'...")
    edges = []

    groups = df.groupby('disease_family').groups
    print(f"Number of disease families: {len(groups)}")

    for group_name, group in groups.items():
        group_indices = group.values  # Directly access group values
        num_nodes = len(group_indices)

        if num_nodes <= 1:
            print(f"Group '{group_name}' has only {num_nodes} node(s). Skipping.")
            continue  # No edges can be formed

        # Determine the actual number of neighbors
        actual_k = min(k, num_nodes - 1)

        # Shuffle the indices for randomness
        shuffled_indices = np.random.permutation(group_indices)

        # Assign the first 'actual_k' indices as neighbors
        neighbors = np.tile(shuffled_indices[:actual_k], (num_nodes, 1))

        # Create source and target edges
        source_nodes = np.repeat(group_indices, actual_k)
        target_nodes = neighbors.flatten()

        # Append to edge list
        edges.extend(zip(source_nodes, target_nodes))

    # Convert to NumPy array
    edges = np.array(edges)

    print("Removing duplicate edges and self-loops...")
    # Remove self-loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]

    # Remove duplicate edges
    edges = np.unique(edges, axis=0)
    print(f"Total edges after sampling: {len(edges)}")

    if len(edges) == 0:
        raise ValueError("No edges were created. Please check the edge creation logic.")

    # Convert to PyTorch tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    return edge_index

# Load model and compute embeddings
def load_model_and_compute_embeddings(
    data: Data,
    model_path: str,
    device: torch.device
) -> torch.Tensor:
    """
    Load the saved model and compute embeddings for the given data.

    Args:
        data (Data): PyTorch Geometric Data object.
        model_path (str): Path to the saved model (.pth file).
        device (torch.device): Device to load the model on.

    Returns:
        torch.Tensor: Embeddings of shape [num_users, embedding_dim].
    """
    # Initialize the model
    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = 32
    heads = 4
    model = PatientGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads
    )

    # Load the model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Compute embeddings
    with torch.no_grad():
        embeddings = model(data.x.to(device), data.edge_index.to(device))

    embeddings = embeddings.cpu()
    print("Embeddings computed.")

    return embeddings

# Function to get recommendations based on embeddings
def get_recommendations(
    embeddings: torch.Tensor,
    user_id: int,
    top_k: int = 3
) -> List[int]:
    """
    Get top_k recommendations for a given user based on cosine similarity of embeddings.

    Args:
        embeddings (torch.Tensor): Tensor of shape [num_users, embedding_dim].
        user_id (int): ID of the user to get recommendations for.
        top_k (int): Number of recommendations to return.

    Returns:
        List[int]: List of recommended user IDs.
    """
    if user_id >= embeddings.size(0):
        raise ValueError(f"user_id {user_id} is out of range.")

    user_embedding = embeddings[user_id].unsqueeze(0)
    similarities = torch.cosine_similarity(user_embedding, embeddings)
    recommended_users = similarities.argsort(descending=True)[1:top_k+1].tolist()  # Skip self
    return recommended_users

# Preload data and model at startup
def initialize(
    filepath: str = 'SynDisNet.csv',
    sample_size: int = 50,
    k: int = 10,
    model_path: str = 'best_model.pth'
) -> (torch.Tensor, pd.DataFrame, Dict[str, LabelEncoder]):
    """
    Initialize data, load model, and compute embeddings.

    Args:
        filepath (str): Path to the CSV dataset.
        sample_size (int): Number of users to load.
        k (int): Number of neighbors per node.
        model_path (str): Path to the saved model.

    Returns:
        torch.Tensor: Embeddings of sampled users.
        pd.DataFrame: Sampled DataFrame.
        Dict[str, LabelEncoder]: Encoders used for categorical features.
    """
    # Load pre-fitted encoders and scaler
    try:
        encoders = joblib.load('encoders.joblib')
        scaler = joblib.load('scaler.joblib')
        print("Loaded pre-fitted encoders and scaler.")
    except FileNotFoundError:
        print("Pre-fitted encoders and scaler not found. They will be created now.")
        encoders = None
        scaler = None

    # Load and preprocess data
    data, encoders, scaler, df_sampled = load_preprocess_data(
        filepath=filepath,
        sample_size=sample_size,
        k=k,
        encoders=encoders,
        scaler=scaler
    )

    # Save encoders and scaler for future use
    joblib.dump(encoders, 'encoders.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Saved encoders and scaler for future use.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and compute embeddings
    embeddings = load_model_and_compute_embeddings(
        data=data,
        model_path=model_path,
        device=device
    )

    return embeddings, df_sampled, encoders

# Initialize embeddings and data at startup
embeddings, df_sampled, encoders = initialize()

# WebSocket endpoint for real-time recommendations
@app.websocket("/ws/recommendations")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Select a random user from the sampled set
            user_id = random.randint(0, embeddings.size(0) - 1)
            recommendations = get_recommendations(embeddings, user_id, top_k=3)

            # Retrieve user information
            user_info = df_sampled.iloc[user_id].to_dict()

            # Decode disease_family and disease for the main user
            if 'disease_family' in encoders and 'disease' in encoders:
                try:
                    user_disease_family = encoders['disease_family'].inverse_transform([user_info.get("disease_family")])[0]
                except ValueError:
                    user_disease_family = "Unknown"

                try:
                    user_disease = encoders['disease'].inverse_transform([user_info.get("disease")])[0]
                except ValueError:
                    user_disease = "Unknown"
            else:
                user_disease_family = "Unknown"
                user_disease = "Unknown"

            # Retrieve and decode recommended users' information
            recommended_users_info = []
            for rec_id in recommendations:
                rec_info = df_sampled.iloc[rec_id].to_dict()
                if 'disease_family' in encoders and 'disease' in encoders:
                    try:
                        rec_disease_family = encoders['disease_family'].inverse_transform([rec_info.get("disease_family")])[0]
                    except ValueError:
                        rec_disease_family = "Unknown"

                    try:
                        rec_disease = encoders['disease'].inverse_transform([rec_info.get("disease")])[0]
                    except ValueError:
                        rec_disease = "Unknown"
                else:
                    rec_disease_family = "Unknown"
                    rec_disease = "Unknown"

                recommended_users_info.append({
                    "user_id": int(rec_id),
                    "disease": rec_disease,
                    "disease_family": rec_disease_family
                })

            # Prepare the response with decoded disease_family and disease
            response = {
                "user_id": int(user_id),
                "user_info": {
                    "disease": user_disease,
                    "disease_family": user_disease_family
                },
                "recommendations": recommended_users_info
            }

            # Send data to frontend
            await websocket.send_json(response)
            await asyncio.sleep(0.5)  # Adjust for update frequency
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        await websocket.close()
