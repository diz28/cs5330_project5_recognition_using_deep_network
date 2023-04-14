# Di Zhang
# Mar 31, 2023
# CS5330 - Computer Vision

import torch
import torch.nn as nn
import torchvision
from main import Net
import torch.nn.functional as F
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from main import test
import os


# greek letter data set transform
class GreekTransform:
    # initiate the transform
    def __init__(self):
        pass

    # actually implementation of the transform
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# import training data for ./greek_train folder
def get_data(batch_size):
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('./greek_train',
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=batch_size,
        shuffle=True)
    return greek_train


# import hand written test greek letter data
def get_data_letters(batch_size):
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('./handwritten_greek_letters',
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=batch_size,
        shuffle=True)
    return greek_train


# pre-processing/threshold handwritten greek letter images
def threshold_handwritten_letters(filename):
    # Define the input and output folders
    input_folder = 'handwritten_greek_letters'

    # Read the input image
    img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

    # Threshold the image using a threshold value of 128
    thresh_value = 128
    ret, thresh = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)
    inv_thresh = cv2.bitwise_not(thresh)
    return inv_thresh


# plot the training and testing loss
def plot(train_counter, train_losses, test_counter, test_losses):
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


# train greek letters dataset
def train_greek_letter(epoch, network, optimizer, train_loader, train_losses, train_counter, log_interval, batch_size):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 6) + ((epoch-1)*len(train_loader.dataset)))


# main method/driver of this class
def main():
    n_epochs = 50
    batch_size = 6
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 1
    random_seed = 42

    # eliminate randomness while developing
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    model = Net()
    print('before: ')
    print(model)
    network_state_dict = torch.load('./results/model.pth')
    model.load_state_dict(network_state_dict)

    # freezes the parameters for the whole network
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f'before Layer {name} is frozen')

    # replace last layer with 3 nodes
    model.fc2 = nn.Linear(50, 3)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f'after Layer {name} is frozen')

    print('after')
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_loader = get_data(batch_size)
    test_loader = get_data_letters(batch_size)
    # Get the shape of the DataLoader
    num_samples = len(train_loader)
    num_batches = len(train_loader)
    batch_shape = next(iter(train_loader))[0].shape

    print("Number of samples:", num_samples)
    print("Number of batches:", num_batches)
    print("Batch shape:", batch_shape)

    data = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(data)

    plt.figure()
    for i in range(len(example_data)):
        # set the plot to be 2 by 3
        plt.subplot(3, 4, i+1)
        # set it to be a tight plot
        plt.tight_layout()
        # set a few parameters
        cur_data = example_data[i][0]
        plt.imshow(cur_data, cmap='gray', interpolation='none')
        plt.title("{}: {}".format('Ground Truth', example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [j*len(train_loader.dataset) for j in range(n_epochs + 1)]
    print(len(train_loader.dataset))

    # evaluate the model first before training
    test(model, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train_greek_letter(epoch, model, optimizer, train_loader, train_losses, train_counter, log_interval, batch_size)
        test(model, test_loader, test_losses)

    print(len(train_counter))
    print(len(train_losses))
    print(len(test_counter))
    print(len(test_losses))
    plot(train_counter, train_losses, test_counter, test_losses)


# the start of the program
if __name__ == '__main__':
    main()


