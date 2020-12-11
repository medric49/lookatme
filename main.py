import cv2
import torch

import config
import models
import utils

if __name__ == '__main__':
    image = cv2.imread('joffrey.jpg')
    image = cv2.resize(image, (config.im_height, config.im_width))

    x = torch.tensor([image, image], dtype=torch.float, device=config.device)
    x = x.view((-1, 3, config.im_height, config.im_width))

    net = models.create_model()
    net.eval()

    output = net(x)
    utils.compute_losses(output)
    cv2.imwrite('test.png', image)
