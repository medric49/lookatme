import torchvision

import config
import models
import torch


class TargetTransform:
    def __call__(self, t):
        return torch.tensor(t, dtype=torch.float, device=config.device)


class ToDevice:
    def __call__(self, pic):
        return pic.to(device=config.device)


class TransposeToCHW:
    def __call__(self, pic):
        return pic.view((3, config.im_height, config.im_width))


def compute_losses(model_output, criterion):
    # (?, 5, 14, 14)

    loss = 0
    for index in range(model_output.shape[0]):
        # (5, 14, 14)
        boxes = build_boxes(model_output[index])
        # (196, 5)

        boxes = confident_boxes(boxes)
        # (? , 5)

        boxes = nms_filter(boxes)
        # (?, 5)

        loss += criterion(boxes, torch.tensor([1, 85, 70, 170, 150], dtype=torch.float, device=config.device))

    return loss.mean()


def build_boxes(x):
    # (5, 14, 14)
    for i in range(models.NB_H_BB):
        for j in range(models.NB_W_BB):
            x0 = i * models.BB_HEIGHT + x[1, i, j] * models.BB_HEIGHT
            y0 = j * models.BB_WIDTH + x[2, i, j] * models.BB_WIDTH
            h = x[3, i, j] * models.BB_HEIGHT
            w = x[4, i, j] * models.BB_WIDTH

            x[1, i, j] = max(x0 - h / 2, 0)  # x1
            x[2, i, j] = max(y0 - w / 2, 0)  # y1
            x[3, i, j] = min(x0 + h / 2, config.im_height)  # x2
            x[4, i, j] = min(y0 + w / 2, config.im_width)  # y2

    x = x.view(5, -1)
    # (5, 196)

    x = x.transpose(1, 0)
    # (196, 5)
    return x


def confident_boxes(x):
    # (196, 5)
    x = x[x[:, 0] > config.detection_threshold, :]
    # (?, 5)
    return x


def nms_filter(x):
    boxes, scores = x[:, 1:], x[:, 0]
    bounding_boxes_indices = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=config.iou_threshold)
    return x[bounding_boxes_indices]


