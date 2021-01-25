import torch.utils.data
import torchvision.datasets

import models
import utils
from data import *


def train():
    images, targets = load_face_data()
    datasets = torch.utils.data.TensorDataset(images, targets)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=config.batch_size, shuffle=True)
    print('Data loaded... ')
    datasets_len = len(datasets)

    net = models.OverFeatRegressionModel(config.overfeat_state_file, config.overfeat_regression_state_file,
                                         overfeat_training=False).to(device=config.device)

    criterion = models.Loss()
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01)

    net.train()
    min_loss = np.inf
    for epoch in range(config.epochs):
        print(f'#### Epoch {epoch} #####')

        running_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device=config.device), labels.to(device=config.device)
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        running_loss /= datasets_len
        print(f'Loss: {running_loss}')
        if running_loss < min_loss:
            print('*** save ***')
            net.save(config.overfeat_regression_state_file)
            min_loss = running_loss


def test():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        utils.ToDevice()
    ])

    target_transform = torchvision.transforms.Compose([
        utils.TransformToLong(),
        utils.ToDevice()
    ])

    datasets = torchvision.datasets.ImageFolder(config.images_folder, transform=transform,
                                                target_transform=target_transform)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=config.batch_size, shuffle=True)

    net = models.OverFeatClassificationModel(config.overfeat_state_file, config.overfeat_classifier_state_file, overfeat_training=False).to(device=config.device)

    net.eval()

    inputs, labels = next(iter(dataloader))
    outputs = torch.softmax(net(inputs), dim=1).argmax(dim=1)
    print(outputs)
    print(labels)

    error = torch.mean(1. * (outputs == labels) )

    print('Accuracy : ', error.item())


def test_image(image_file, output_file):
    net = models.OverFeatRegressionModel(config.overfeat_state_file, config.overfeat_regression_state_file,
                                             overfeat_training=False).to(device=config.device)

    net.eval()

    image = cv2.imread(image_file)
    image = cv2.resize(image, (config.im_width, config.im_height))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        utils.ToDevice()
    ])

    x = transform(image)

    x = x.view((-1, 3, config.im_height, config.im_width))

    output = net(x)[0]
    print(output)

    point_1 = int(output[2].item() * config.im_height), int(output[1].item() * config.im_width)
    point_2 = int(output[4].item() * config.im_height), int(output[3].item() * config.im_width)
    cv2.rectangle(image, point_1, point_2, (0, 0, 255, 10), 2)
    cv2.imwrite(output_file, image)



if __name__ == '__main__':
    # train()
    # test()
    test_image('ia_data/faces_tmp/24859.png', 'demo.png')

