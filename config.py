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

epochs = 125
dropout_prob = 0

batch_size = 64

images_folder = 'images'

overfeat_state_file = 'overfeat_state.pt'
overfeat_classifier_state_file = 'overfeat_classifier.pt'

