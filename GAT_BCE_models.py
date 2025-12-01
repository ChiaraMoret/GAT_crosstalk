import sys
import os
import time
import datetime
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MiGe_dataset import *
# -----------------------------
# Model Definition
# -----------------------------

def split_attention(att, batch):
    """
    Split batched attention into per-graph attention.
    Used in the model
    
    Args:
        att (tuple): (edge_index, att_weights)
        batch (LongTensor): Node-to-graph assignment [num_nodes]

    Returns:
        List[Tuple[edge_index, att_weights]] per graph
    """
    edge_index, att_weights = att
    num_graphs = batch.max().item() + 1
    num_nodes_per_graph = torch.bincount(batch)

    attn_per_graph = []
    node_offset = 0

    # graph assignment of source nodes
    edge_graph = batch[edge_index[0]]

    for g in range(num_graphs):
        mask = (edge_graph == g)  # edges belonging to graph g
        ei_g = edge_index[:, mask] - node_offset  # shift node indices
        aw_g = att_weights[mask]
        attn_per_graph.append((ei_g, aw_g))
        node_offset += num_nodes_per_graph[g]

    return attn_per_graph

    
class GAT_BCE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, heads=1, dropout=0.0):
        super(GAT_BCE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(self.num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))

        self.lin = Linear(hidden_channels * heads, 1)

    def forward(self, x, edge_index, batch, return_attention=False, return_embedding=False):
        attentions = {}
        embeddings = {}
        
        for i, conv in enumerate(self.convs):
            if return_attention:
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attentions[f'layer_{i}'] = split_attention(attn, batch)
                # embeddings[f'layer_{i}'] = x
            else: 
                x = conv(x, edge_index)
            

            x = F.elu(x)
            
            if return_embedding:
                embeddings[f'layer_{i}'] = x

            
    
        # Readout
        x = global_mean_pool(x, batch)
        
        # Dropout
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embedding:
            embeddings['final'] = x
        x = self.lin(x)  # [batch_size, 1]
    
        # ---- Build return dict ----
        if return_attention and return_embedding:
            return x.squeeze(-1), attentions, embeddings

        if return_attention:
            return x.squeeze(-1), attentions

        if return_embedding:
            return x.squeeze(-1), embeddings

        return x.squeeze(-1)


# -----------------------------
# Training and Evaluation
# -----------------------------
def train(model, train_loader, optimizer, criterion, device, l1_lambda=0.0):
    model.train()
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        labels = data.y.float()
        loss = criterion(out, labels)

        # ---- L1 regularization (minimal change) ----
        if l1_lambda > 0:
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_penalty
        # -------------------------------------------

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.num_graphs
    return train_loss / len(train_loader.dataset)


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        labels = data.y.float()
        loss = criterion(out, labels)
        val_loss += loss.item() * data.num_graphs
    return val_loss / len(val_loader.dataset)

# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def test(model, loader, device):
    model.eval()
    all_preds = []
    all_true = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()
        all_preds.append(preds)
        all_true.append(data.y.float())

    all_preds = torch.cat(all_preds).cpu()
    all_true = torch.cat(all_true).cpu()

    f1 = f1_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, zero_division=0)
    recall = recall_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)

    return f1, precision, recall, cm


@torch.no_grad()
def inference(model, val_loader, device, return_attention=True):
    model.eval()

    all_outs = []
    all_atts = {f'layer_{i}': [] for i in range(model.num_layers)} if return_attention else None
    
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, return_attention=return_attention)
        
        if return_attention:
            out, atts = out
            for layer in all_atts.keys():
                all_atts[layer].append(atts[layer])
        
        all_outs.append(out)

    # flatten
    all_outs = torch.cat(all_outs, dim=0)
    
    if return_attention:
        for layer in all_atts:
            all_atts[layer] = [x for sublist in all_atts[layer] for x in sublist]
        return all_outs, all_atts
        
    return all_outs
