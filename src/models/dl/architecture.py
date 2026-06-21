import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTemporalCNN(nn.Module):
    """
    SpatialTemporalCNN (STCNN): Applies a learnable dense spatial filter (channels x channels matrix)
    followed by a 1D temporal convolution. This is a CNN with learned spatial mixing.
    """

    def __init__(
        self,
        channels=19,
        temp_filters=32,
        kernel_size=16,
        n_layers=1,
        adj_init="identity",
        adj_norm="sigmoid",
        dropout=0.4,
        classes=2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.adj_norm_type = adj_norm

        if adj_init == "uniform":
            init_w = torch.ones(channels, channels) / channels
        elif adj_init == "identity":
            init_w = torch.eye(channels)
        else:
            init_w = torch.randn(channels, channels) * 0.1
        self.adj = nn.Parameter(init_w)

        self.conv1 = nn.Conv1d(
            channels, temp_filters, kernel_size=kernel_size, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(temp_filters)

        if n_layers == 2:
            self.conv2 = nn.Conv1d(
                temp_filters, temp_filters, kernel_size=kernel_size, padding="same"
            )
            self.bn2 = nn.BatchNorm1d(temp_filters)
            self.res_proj = nn.Conv1d(channels, temp_filters, kernel_size=1, bias=False)

        self.pool = nn.AdaptiveAvgPool1d(8)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(temp_filters * 8, classes)

    def _apply_spatial_filter(self, x):
        if self.adj_norm_type == "softmax":
            adj = F.softmax(self.adj, dim=-1)
        elif self.adj_norm_type == "sigmoid":
            adj = torch.sigmoid(self.adj)
        else:
            adj = self.adj
        return (x.transpose(1, 2) @ adj).transpose(1, 2)

    def forward(self, x):
        x_s = self._apply_spatial_filter(x)
        out = F.elu(self.bn1(self.conv1(x_s)))
        if self.n_layers == 2:
            residual = self.res_proj(x_s)
            out = F.elu(self.bn2(self.conv2(out)) + residual)
        out = self.pool(out)
        return self.fc(self.dropout(out.flatten(1)))
