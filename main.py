import cv2
import torch

import config
import models
import utils
import torch.optim as optim

if __name__ == '__main__':
    image = cv2.imread('joffrey.jpg')
    image = cv2.resize(image, (config.im_height, config.im_width))

    x = torch.tensor([image, image, image, image, image, image, ], dtype=torch.float, device=config.device)
    x = x.view((-1, 3, config.im_height, config.im_width))

    # torch.autograd.set_detect_anomaly(True)

    net = models.create_model()
    criterion = models.Loss()
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, betas=config.betas)

    print(f'### TRAINING ###')
    net.train()
    for epoch in range(config.epochs):
        print(f'### epoch {epoch} ###')
        optimizer.zero_grad()
        output = net(x)
        loss = utils.compute_losses(output, criterion)
        print(loss.item())
        loss.backward()
        optimizer.step()

    print(f'### EVALUATION ###')
    net.eval()

    x = x[0:1]
    output = net(x)[0]
    boxes = utils.build_boxes(output)
    boxes = utils.confident_boxes(boxes)
    boxes = utils.nms_filter(boxes)

    box = boxes[0]
    print(box)
    cv2.rectangle(image, (int(box[2]), int(box[1])), (int(box[4]), int(box[3])), (0, 0, 255, 10), 2)
    cv2.imwrite('test.png', image)



