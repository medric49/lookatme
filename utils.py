import torch
import torchvision

import config


def compute_losses(model_output):
    # (?, 5, 14, 14)
    for x in model_output:
        # (5, 14, 14)
        x = nms(x)


def nms(x):
    # (5, 14, 14)
    x = x.view(5, -1)
    # (5, 196)

    x = x.transpose(0, 1)
    # (196, 5)

    x = x[x[:, 0] > 0.5, :]

    boxes, scores = x[:, 1:], x[:, 0]
    output = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=config.iou_threshold)
    return output


