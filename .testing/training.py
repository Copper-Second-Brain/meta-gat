import torch
from torch_geometric.data import Data
from meta_gat_model import MetaGAT

# Define constants for data
num_nodes = 100  # Number of nodes
num_edges = 300  # Number of edges
node_feature_dim = 512  # Ensure this is 512, matching model expectations
edge_feature_dim = 128  # Edge features

# Generate synthetic node features
x = torch.randn((num_nodes, node_feature_dim))  # Each node has 512 features

# Randomly generated edges and edge features
edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
edge_attr = torch.randn((num_edges, edge_feature_dim))  # Edge feature dimension of 128

# Random binary labels for node classification
y = torch.randint(0, 2, (num_nodes,), dtype=torch.long)

# Create a Data object for the graph
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# Initialize model with in_channels matching node feature dimensions
model = MetaGAT(in_channels=node_feature_dim, hidden_channels=64, num_classes=2, num_layers=2, heads=8, dropout=0.6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train(model, data, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Train the model
train(model, graph_data)
