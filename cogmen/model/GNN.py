import torch.nn as nn
from torch_geometric.nn import  RGATConv


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GNN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGATConv(g_dim, h1_dim, self.num_relations)
        self.conv2 = RGATConv(h1_dim, h2_dim, self.num_relations)
        self.bn = nn.BatchNorm1d(h2_dim * args.gnn_nheads)

    def forward(self, node_features, edge_index, edge_type):
        x = self.conv1(node_features, edge_index, edge_type).relu()
        x = nn.functional.leaky_relu(self.bn(self.conv2(x, edge_index, edge_type)))

        return x
