import torch
import torch.nn.functional as F
import networkx as nx
import pickle
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
from pytorch_tabnet import tab_network
from torch.nn.utils import clip_grad_norm_

#GNN model
class GCNConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(GCNConvBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.dropout(x)
        return x
    
class GNNNet(torch.nn.Module):
    def __init__(self, in_channels, embedding_dim, out_channels, dropout_rate, number_of_layers):
        super(GNNNet, self).__init__()
        self.embedding_layer = GCNConv(in_channels, embedding_dim)
        self.convs = torch.nn.ModuleList(
            [GCNConvBlock(embedding_dim, embedding_dim, dropout_rate) for i in range(number_of_layers)]
            )
        self.output_layer = GCNConv(embedding_dim, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embedding_layer(x, edge_index, edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr) + x
        #intermediate_layer = x
        x = self.output_layer(x, edge_index, edge_attr)
        return x[batch,:]#, intermediate_layer


# Time series model
import torch
from torch import nn

class TimeStampEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=445+1):

        '''
        0 is the padding index
        '''

        super(TimeStampEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.te = torch.nn.Embedding(max_len, d_model)
        # Q: is self.te trainable?
        # A: yes, it is trainable
        # Q: why is it trainable?
        # A: because it is a nn.Embedding object, which is a trainable layer
        # Q: register_buffer is not trainable?
        # A: register_buffer is not trainable, it is a tensor


    def forward(self, x, timestamp):
        x = x + self.te(timestamp)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, nout = 12, ninp = 4, nhead = 4, nhid = 8, nlayers = 3, dropout=0.15):
        super(TransformerBlock, self).__init__()
        self.project_layer = nn.Linear(ninp, nhid)
        self.pos_encoder = TimeStampEmbedding(nhid)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=nhid, dropout=dropout),
            num_layers=nlayers
        )
        self.decoder = nn.Linear(nhid, nout)
        self.avgpool = torch.nn.AvgPool1d(kernel_size = 445, stride = 1)
        self.output_layer = nn.Linear(nout,16)

    def generate_square_subsequent_mask(self, sz):
        
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask

    def forward(self, src, src_mask, timestamp):
        src = self.project_layer(src)
        src = self.pos_encoder(src, timestamp)
        tgt_mask = self.generate_square_subsequent_mask(src.size(1)).cuda()
        src = src.permute(1,0,2)
        output = self.transformer_decoder(src, src,
                                          tgt_mask = tgt_mask, memory_mask = tgt_mask,
                                          tgt_key_padding_mask = src_mask, memory_key_padding_mask  = src_mask)
        output = output.permute(1,0,2)
        output = self.decoder(output)
        output = self.avgpool(output.permute(0,2,1)).squeeze()
        output = self.output_layer(output)
        return output

# mixture of experts model
class HTGTM(nn.Module):
    def __init__(self, gnnmodel, tsmodel, tabmodel):
        super().__init__()

        self.gnnmodel = gnnmodel # output_dim = 16
        self.tsmodel = tsmodel # output_dim = 16
        self.tabmodel = tabmodel # output_dim = 16

        self.gate = nn.Sequential(
            nn.Linear(16*3, 16*3),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(0.15)

        self.out_project = nn.Linear(16*3, 16)

        self.output_layer = nn.Linear(16, 1)

    def forward(self, features, node_features, edge_index, edge_attr, batch, tsfeature, tstimestamp, tsmask):
        gnn_output = self.gnnmodel(node_features, edge_index, edge_attr, batch)
        ts_output = self.tsmodel(tsfeature, tsmask, tstimestamp)
        tab_output,M_loss = self.tabmodel(features)
        #print(gnn_output.shape, ts_output.shape, tab_output.shape)

        output = torch.cat([gnn_output, ts_output, tab_output], dim=1)
        output = self.dropout(output)
        gate = self.gate(output)
        scored_output = gate * output
        scored_output = self.out_project(scored_output)
        output = self.output_layer(scored_output)
        return output, M_loss