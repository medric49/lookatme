import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
gpu = torch.device('cuda')

detection_threshold = 0.5
iou_threshold = 0.6

im_height = 231
im_width = 231

learning_rate = 0.05
betas = (0.9, 0.999)

epochs = 10
dropout_prob = 0.5

batch_size = 128

images_folder = 'images'

net_state_file = 'state.pth'


