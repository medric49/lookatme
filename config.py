import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
gpu = torch.device('cuda')

iou_threshold = 0.6

im_width = 224
im_height = 224
