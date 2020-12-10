import torch.nn as nn


def default_model():

    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3)),
        # (224, 224, 3)
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        # (112, 112, 64)
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        # (56, 56, 128)
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        # (28, 28, 256)
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        # (14, 14, 512)
        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), bias=True),
        nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), bias=True),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), bias=True),
        nn.Conv2d(in_channels=16, out_channels=5, kernel_size=(1, 1), bias=True),
        # (14, 14, 5)
    )
