import torch
import torchvision

# from pytorchcv.model_provider import get_model
from torchvision import models
import torch.nn as nn
import math
from torch.nn.modules.utils import _triple
import torch.nn.functional as F
from torch.autograd import Variable


def PositionalEncoding(pos, n_dim):
    batch = len(pos)
    pe = torch.zeros(batch, n_dim)
    for b in range(batch):
        for i in range(0, n_dim, 2):
            pe[b, i] = math.sin(pos[b] / (10000 ** ((2 * i) / n_dim)))
            pe[b, i + 1] = math.cos(pos[b] / (10000 ** ((2 * (i + 1)) / n_dim)))
    return pe.to("cuda:0")


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # resnet = CNN(num_classes=[19,0,0], mode='subtask')
        # resnet.load_state_dict(torch.load('./runs/Resnet50_ASAP.pth'))
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
        )

    def forward(self, x):
        # with torch.no_grad():
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


class Attention(nn.Module):
    def __init__(self, latent_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.latent_attention = nn.Linear(latent_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.joint_attention = nn.Linear(attention_dim, 1)

    def forward(self, latent_repr, hidden_repr):
        if hidden_repr is None:
            hidden_repr = [
                Variable(
                    torch.zeros(
                        latent_repr.size(0), 1, self.hidden_attention.in_features
                    ),
                    requires_grad=False,
                ).float()
            ]
        h_t = hidden_repr[0]
        latent_att = self.latent_attention(latent_att)
        hidden_att = self.hidden_attention(h_t)
        joint_att = self.joint_attention(F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w


class ConvLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
        mode=None,
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            # nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.attention_layer = nn.Linear(
            2 * hidden_dim if bidirectional else hidden_dim, 1
        )
        self.mode = mode

    def forward(self, x, frame_num):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)

        x = x.view(batch_size, seq_length, -1)
        ## add the positional encoding
        x = x + frame_num
        x = self.lstm(x)
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers(x)
