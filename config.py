import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
gpu = torch.device('cuda')

detection_threshold = 0.5
iou_threshold = 0.6

im_height = 224
im_width = 224

learning_rate = 0.002
betas = (0.9, 0.999)

epochs = 50

