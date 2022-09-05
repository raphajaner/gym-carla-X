import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torchvision.models import resnet18


class GraphEncoder(torch.nn.Module):
    def __init__(self, gnn_depth, edge_geo_dim, map_feat_dim, edge_dim, node_dim, msg_dim,
                        rnn_input_dim, rnn_hidden_dim, rnn_layers, rnn_do):
        super(GraphEncoder, self).__init__()
        """Initialize the network architecture
        Args:
            input_dim ([int]): [Number of time lags considered]
            hidden_dim ([int]): [The dimension of RNN output]
            num_layers (int, optional): [Number of stacked RNN layers]. Defaults to 1.
            do (float, optional): [Dropout for regularization]. Defaults to .05.
        """
        self.depth = gnn_depth
        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.rnn_do = self.rnn_do

        self.map_channels = 3

        # node feature encoding
        self.curr_actor_encoder = nn.Sequential(
            nn.Linear(5, 12),
            nn.ReLU(),
            nn.Linear(12, 24)
        )

        self.node_encoder = nn.Sequential(
            nn.Linear(24+map_feat_dim+rnn_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 48)
        )

        self.history_encoder = nn.LSTM(input_size=rnn_input_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_layers, dropout=rnn_do)
        self.fc1 = nn.Linear(in_features=rnn_hidden_dim, out_features=int(rnn_hidden_dim / 2))
        self.act1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_features=int(rnn_hidden_dim / 2))

        self.map_encoder = resnet18(pretrained=False)
        self.map_encoder.conv1 = nn.Conv2d(self.map_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.map_encoder.fc = nn.Linear(self.map_encoder.fc.in_features, map_feat_dim)

        self.actor_message_passing = MessagePassing(aggr="add")

    def forward(self, data):
        curr_actor_feats, actor_hist, actor_maps, edge_attr, edge_index, node_gt, edge_gt, batch = (
            data.curr_actor_feats,
            data.actor_hist,
            data.map_feats,
            data.edge_attr,
            data.edge_index,
            data.node_gt,
            data.edge_gt,
            data.batch_idx
        )

        # Encoding actor history
        hidden_state = torch.zeros(self.rnn_layers, actor_hist.shape[1], self.rnn_hidden_dim).to(self.device)
        cell_state = hidden_state
        history_feat, _ = self.history_encoder(actor_hist, (hidden_state, cell_state))

        # Encoding current actor feats
        actor_feat = self.actor_feat_encoder(curr_actor_feats.float())
        map_feat = self.map_encoder(actor_maps)

        # Combine encoded node data
        x = torch.cat([actor_feat, history_feat, map_feat], dim=1)  # E x (D_E1+D_E2)
        initial_x = x

        edge_attr = self.edge_encoder(edge_attr.float())

        for i in range(self.depth):
            x, edge_attr = self.message_passing.forward(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                                        initial_x=initial_x)



        return self.edge_classifier(edge_attr), self.node_classifier(x)