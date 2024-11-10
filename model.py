import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# Hyperparameters
embedding_dim = 128
attention_heads = 8
learning_rate = 0.001
epochs = 200

# Define the Graph Neural Network with Multi-Head Attention
class RecommendationGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(RecommendationGNN, self).__init__()
        
        # Graph Convolutional Layer
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        
        # Multi-Head Attention Layer
        self.attention = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        
        # Final Layer to output embeddings for recommendations
        self.gcn2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        # Pass through GCN layer
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        
        # Pass through Multi-Head Attention layer
        x = self.attention(x, edge_index)
        x = torch.relu(x)
        
        # Final GCN layer for output
        x = self.gcn2(x, edge_index)
        return x

# Define loss and optimization function
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Use only the positive and negative edges for loss computation
    pos_out = (out[data.train_pos_edge_index[0]], out[data.train_pos_edge_index[1]])
    neg_out = (out[data.train_neg_edge_index[0]], out[data.train_neg_edge_index[1]])
    
    # Calculate loss
    pos_loss = -torch.log(torch.sigmoid((pos_out[0] * pos_out[1]).sum(dim=-1) + 1e-15)).mean()
    neg_loss = -torch.log(1 - torch.sigmoid((neg_out[0] * neg_out[1]).sum(dim=-1) + 1e-15)).mean()
    loss = pos_loss + neg_loss
    
    loss.backward()
    optimizer.step()
    return loss.item()

# Example: Generating sample data (nodes and edges)
def create_sample_data(num_users):
    # Randomly generate node features for each user
    x = torch.randn((num_users, embedding_dim))
    
    # Random edge list (for training purposes, normally use real interaction data)
    edge_index = torch.randint(0, num_users, (2, 2 * num_users))  # Example random edges
    
    data = Data(x=x, edge_index=edge_index)
    
    # Define train_pos_edge_index as the original edge_index for positive samples
    data.train_pos_edge_index = data.edge_index
    
    # Generate negative samples for edges
    data.train_neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=num_users, num_neg_samples=data.edge_index.size(1))
    
    return data

# Initialize model, data, and optimizer
num_users = 100  # Example number of users
data = create_sample_data(num_users)

model = RecommendationGNN(in_channels=embedding_dim, hidden_channels=embedding_dim, out_channels=embedding_dim, heads=attention_heads)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    loss = train(model, data, optimizer, None)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Generating Recommendations
def recommend_users(model, data, user_id, top_k=5):
    model.eval()
    with torch.no_grad():
        # Get embeddings
        embeddings = model(data.x, data.edge_index)
        
        # Cosine similarity for recommendations
        user_embedding = embeddings[user_id].unsqueeze(0)
        similarities = torch.cosine_similarity(user_embedding, embeddings)
        
        # Sort and get top-k recommendations excluding the user
        recommended_users = similarities.argsort(descending=True)[1:top_k+1]
        return recommended_users

# Example: Get recommendations for user 0
user_id = 0
recommended_users = recommend_users(model, data, user_id, top_k=5)
print(f"Recommended users for user {user_id}: {recommended_users.tolist()}")
