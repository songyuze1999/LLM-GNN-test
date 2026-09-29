"""Graph anomaly and root-cause models used by Phases 10-12."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, global_mean_pool


class GraphAnomalyRCA(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, heads: int, layers: int, dropout: float, kind: str) -> None:
        super().__init__()
        if kind not in {"gcn", "gat", "gatv2"}:
            raise ValueError(f"unsupported graph model: {kind}")
        self.kind = kind
        self.dropout = float(dropout)
        self.projection = nn.Linear(input_size, hidden_size)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(int(layers)):
            in_size = hidden_size
            if kind == "gcn":
                conv = GCNConv(in_size, hidden_size)
            elif kind == "gat":
                conv = GATConv(in_size, hidden_size, heads=heads, concat=False, dropout=dropout)
            else:
                conv = GATv2Conv(in_size, hidden_size, heads=heads, concat=False, dropout=dropout)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_size))
        self.anomaly_head = nn.Linear(hidden_size, 1)
        self.rca_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, return_attention: bool = False):
        x = torch.relu(self.projection(x))
        attentions = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attention and self.kind == "gatv2":
                x, (_, weights) = conv(x, edge_index, return_attention_weights=True)
                attentions.append(weights.detach())
            else:
                x = conv(x, edge_index)
            x = norm(torch.relu(x))
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        graph_x = global_mean_pool(x, batch)
        return self.anomaly_head(graph_x).view(-1), self.rca_head(x).view(-1), attentions
