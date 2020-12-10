import torch.nn as nn
import torch

import config


class SigRelu(nn.Module):
    def __init__(self):
        super(SigRelu, self).__init__()

    def forward(self, x):
        a = torch.sigmoid(x[:, 0:1, :, :])
        b = torch.relu(x[:, 1:, :, :])
        x[:, 0:1, :, :] = a
        x[:, 1:, :, :] = b


        return x


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
        SigRelu(),
        # (?, 5, 14, 14)
    ).to(device=config.device)
