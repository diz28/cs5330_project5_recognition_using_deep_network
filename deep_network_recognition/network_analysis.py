# Di Zhang
# Mar 31, 2023
# CS5330 - Computer Vision

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from main import Net, get_data


# plot the 10 weights of the first layer of the network
def plot_layer(data):
    plt.figure()
    for i, layer_weight in enumerate(data):
        plt.subplot(3, 4, i + 1)
        plt.imshow(data[i][0])
        plt.title('filter {}'.format(i + 1))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# apply the weights to images and show the output and weights in comparsion
def filter2d(filter_weights):
    batch_size_train = 64
    batch_size_test = 1000
    training_size = 60000

    train_loader, test_loader = get_data(batch_size_train, batch_size_test, training_size)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    image = example_data[0].numpy()

    print(filter_weights.shape)
    with torch.no_grad():
        j = 1
        for i in range(filter_weights.shape[0]):
            kernal = filter_weights[i][0].astype(np.float32)
            plt.subplot(5, 4, j)
            plt.imshow(kernal, cmap='gray', interpolation='none')
            plt.title('filter {}'.format(i))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(5, 4, (i + 1) * 2)
            output = cv2.filter2D(image, -1, kernal)
            output = output.reshape([28, 28])
            plt.imshow(output, cmap='gray', interpolation='none')
            plt.title('output {}'.format(i))
            plt.xticks([])
            plt.yticks([])
            j += 2
        plt.show()


# main/driver method of this class
def main():
    model = Net()
    first_layer_weight = model.conv1.weight.detach().numpy()
    plot_layer(first_layer_weight)
    filter2d(first_layer_weight)
    print(first_layer_weight)


# start of the program
if __name__ == '__main__':
    main()
