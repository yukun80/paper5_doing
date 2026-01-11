# -*- coding: utf-8 -*-
"""
Module: GCN Model Architecture
Location: experiments/02_dl_gcn/gcn_model.py
Description: 
    Implements a standard 2-layer Graph Convolutional Network (Kipf & Welling, 2017).
    Uses PyTorch sparse matrix multiplication (torch.sparse.mm) for efficiency.
    
    Structure:
    Input (N, F) -> GCNConv -> ReLU -> Dropout -> GCNConv -> LogSoftmax (N, 2)

Author: AI Assistant (Virgo Edition)
Date: 2026-01-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Operation: H' = A * H * W + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weights (F_in, F_out)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier Initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_feat: torch.Tensor, adj_matrix: torch.sparse.Tensor) -> torch.Tensor:
        """
        Args:
            input_feat: Node features (N, in_features)
            adj_matrix: Normalized Laplacian (N, N) as SparseTensor
        """
        # 1. Support = Input * Weight
        support = torch.mm(input_feat, self.weight)
        
        # 2. Output = Adj * Support
        output = torch.sparse.mm(adj_matrix, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """
    Standard 2-Layer GCN for Binary Classification.
    """
    def __init__(self, n_feat: int, n_hidden: int, n_class: int = 2, dropout: float = 0.5):
        super(GCN, self).__init__()
        
        self.gc1 = GraphConvolution(n_feat, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_class)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features
            adj: Normalized Adjacency Matrix
        Returns:
            Log probabilities (N, n_class)
        """
        # Layer 1
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Layer 2
        x = self.gc2(x, adj)
        
        # Output
        return F.log_softmax(x, dim=1)