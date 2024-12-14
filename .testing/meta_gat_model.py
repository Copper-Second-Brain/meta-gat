import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import Optional

class MetaAttentionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 8, dropout: float = 0.6):
        super(MetaAttentionLayer, self).__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Meta-learning components with correct dimensions
        self.meta_weight = nn.Parameter(torch.Tensor(heads, in_channels, out_channels))
        self.meta_att = nn.Parameter(torch.Tensor(heads, 2 * out_channels))

        # Domain adaptation layers
        self.domain_projection = nn.Linear(in_channels, out_channels)
        self.domain_attention = nn.Parameter(torch.Tensor(heads, out_channels))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.meta_weight, gain=gain)
        nn.init.xavier_normal_(self.meta_att, gain=gain)
        nn.init.xavier_normal_(self.domain_attention, gain=gain)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, domain_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = x.size(0)

        # Compute domain weights
        if domain_context is not None:
            domain_proj = self.domain_projection(domain_context)
            domain_weights = F.softmax(torch.matmul(domain_proj, self.domain_attention.t()), dim=-1)
        else:
            domain_weights = torch.ones(self.heads, device=x.device) / self.heads

        out = []
        for head in range(self.heads):
            weight = self.meta_weight[head]
            att = self.meta_att[head]

            # Transform input using meta_weight for each head
            x_transformed = torch.matmul(x, weight)  # x: (N, in_channels) * weight: (in_channels, out_channels)
            row, col = edge_index

            # Calculate attention scores
            alpha = torch.cat([x_transformed[row], x_transformed[col]], dim=-1)
            alpha = F.leaky_relu(torch.matmul(alpha, att.unsqueeze(-1))).squeeze(-1)
            alpha = F.softmax(alpha, dim=0)
            alpha = self.dropout(alpha)

            # Accumulate weighted features for each head
            out_head = torch.zeros(N, self.out_channels, device=x.device)
            out_head.index_add_(0, row, alpha.unsqueeze(-1) * x_transformed[col])
            out.append(out_head * domain_weights[head])

        return torch.mean(torch.stack(out), dim=0)

class MetaGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, num_layers: int = 2, heads: int = 8, dropout: float = 0.6, task_type: str = "social_media"):
        super(MetaGAT, self).__init__()
        self.task_type = task_type

        # Input projection layer
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Attention layers
        self.layers = nn.ModuleList([
            MetaAttentionLayer(
                hidden_channels * (heads if i > 0 else 1),  # Adjust in_channels for subsequent layers
                hidden_channels,
                heads=heads,
                dropout=dropout
            ) for i in range(num_layers)
        ])

        # Task-specific head selection
        if task_type == "social_media":
            self.task_head = SocialMediaHead(hidden_channels * heads, num_classes)
        else:
            self.task_head = DiseasePredictionHead(hidden_channels * heads, num_classes)
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        domain_context = getattr(data, 'domain_context', None)

        # Apply input projection
        x = self.input_proj(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Pass through attention layers
        for layer in self.layers:
            x = layer(x, edge_index, domain_context)
            x = F.elu(x)
            x = self.dropout(x)
        
        return self.task_head(x)

class SocialMediaHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(SocialMediaHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_channels // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class DiseasePredictionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(DiseasePredictionHead, self).__init__()
        self.risk_attention = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_channels // 2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attended, _ = self.risk_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x_attended = x_attended.squeeze(0)
        return self.mlp(x_attended)
