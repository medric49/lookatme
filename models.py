import torch.nn as nn
import torch

import config

BB_HEIGHT = 16
BB_WIDTH = 16
NB_H_BB = 14
NB_W_BB = 14


class SigLu(nn.Module):
    def __init__(self):
        super(SigLu, self).__init__()

    def forward(self, x):
        x = torch.relu(x)
        x[:, 0:3, :, :] = torch.sigmoid(x[:, 0:3, :, :])
        # x[:, 3:, :, :] = torch.relu(x[:, 3:, :, :])
        return x


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, x, y):
        # (?, 5)
        z = x - y

        z = z**2

        loss = z[:, 0] + y[0] * (z[:, 1:].sum(dim=1))

        return loss.sum()


def create_model():
    return nn.Sequential(
        # (?, 3, 224, 224)
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        # (?, 64, 112, 112)

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        # (?, 128, 56, 56)

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        # (?, 256, 28, 28)

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
        # (?, 512, 14, 14)

        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), bias=True),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), bias=True),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), bias=True),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=5, kernel_size=(1, 1), bias=True),
        SigLu(),
        # (?, 5, 14, 14)
    ).to(device=config.device)


def create_overfeat_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4)),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.ReLU(),
        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1)),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.ReLU(),
    )


class OverFeatClassModel (nn.Module):
    def __init__(self):
        super(OverFeatClassModel, self).__init__()
        self.overfeat = create_overfeat_model()

        self.l1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(6, 6), stride=(1, 1))
        self.l2 = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.l3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

        self.dropout = nn.Dropout(config.dropout_prob)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.overfeat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.sigmoid(x)

        return x
