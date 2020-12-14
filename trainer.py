import numpy as np
import torch.utils.data
import torchvision.datasets

import config
import models
import utils

if __name__ == '__main__':

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        utils.ToDevice(),
        utils.TransposeToCHW()
    ])

    target_transform = torchvision.transforms.Compose([
        utils.TargetTransform(),
    ])

    datasets = torchvision.datasets.ImageFolder(config.images_folder, transform=transform, target_transform=target_transform)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=config.batch_size, shuffle=True)
    datasets_len = len(datasets)

    net = models.OverFeatClassModel().to(device=config.device)
    try:
        net.load_state_dict(torch.load(config.net_state_file))
    except FileNotFoundError:
        pass

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=net.parameters(), lr=config.learning_rate)

    net.train()
    min_loss = np.inf
    for epoch in range(config.epochs):
        print(f'#### Epoch {epoch} #####')

        running_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()

            outputs = net(images)
            outputs = outputs.view((-1))
            loss = criterion(outputs, labels)

            running_loss += loss.item() * len(images)

            loss.backward()
            optimizer.step()

        running_loss /= datasets_len
        print(f'Loss: {running_loss}')
        if running_loss < min_loss:
            print('*** save ***')
            torch.save(net.state_dict(), config.net_state_file)
            min_loss = running_loss

