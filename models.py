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


def create_vgg16_model():
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
    )


def create_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),
    )


def create_overfeat_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4)),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1)),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    )


class OverFeatClassifierModel (nn.Module):
    def __init__(self, overfeat_state_file=None, state_file=None, overfeat_training=False):

        self.overfeat_training = overfeat_training

        super(OverFeatClassifierModel, self).__init__()
        self.prefix = create_overfeat_model()
        self.in_size = 6*6*1024

        # self.l1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(6, 6), stride=(1, 1))
        # self.l2 = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        # self.l3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

        self.suffix = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

        if overfeat_state_file is not None:
            try:
                self.prefix.load_state_dict(torch.load(overfeat_state_file))
            except FileNotFoundError:
                pass

        if state_file is not None:
            try:
                self.suffix.load_state_dict(torch.load(state_file))
            except FileNotFoundError:
                pass

        if not overfeat_training:
            for param in self.prefix.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.prefix(x)
        x = self.suffix(x)
        return x

    def parameters(self, recurse: bool = True):
        if self.overfeat_training:
            return super(OverFeatClassifierModel, self).parameters()
        else:
            return self.suffix.parameters(recurse)

    def save(self, state_file, overfeat_state_file=None):
        if overfeat_state_file is not None:
            torch.save(self.prefix.state_dict(), overfeat_state_file)

        if state_file is not None:
            torch.save(self.suffix.state_dict(), state_file)
