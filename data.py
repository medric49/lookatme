import os

import cv2
import numpy as np
import torch
import torch.utils.data

import config
import csv


def load_images(image_dir, csv_file, to_array=False):
    nb_file = len(os.listdir(image_dir))

    images = []
    for i in range(nb_file):
        image = f'{image_dir}/{i}.png'

        image = cv2.imread(image)
        image = cv2.resize(image, (config.im_width, config.im_height))

        images.append(image)

    csv_file = open(csv_file)
    csv_reader = csv.reader(csv_file)
    targets = []
    for line in csv_reader:
        targets.append(line)
    csv_file.close()

    targets = targets[1:]
    targets = np.asarray(targets, float).tolist()

    if to_array:
        images, targets = np.array(images), np.array(targets)

    return images, targets


def load_face_data(shuffle=False, to_tensor=True):
    face_images, face_targets = load_images('images/face', 'images/face.csv')
    no_face_images, no_face_targets = load_images('images/no_face', 'images/no_face.csv')

    images = face_images + no_face_images
    targets = face_targets + no_face_targets

    images, targets = np.array(images), np.array(targets)

    if shuffle:
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images, targets = images[indices], targets[indices]

    if to_tensor:
        images = torch.tensor(images.transpose((0, 3, 1, 2)), dtype=torch.float)/255.
        targets = torch.tensor(targets, dtype=torch.float)
    return images, targets


if __name__ == '__main__':
    images, targets = load_face_data()
    datasets = torch.utils.data.TensorDataset(images, targets)
    print(datasets[0])






