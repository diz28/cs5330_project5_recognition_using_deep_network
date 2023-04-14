# Di Zhang
# Mar 31, 2023
# CS5330 - Computer Vision

# import Net main.py
from main import get_data, Net, show_sample

import torch.optim as optim
import torch
import torchvision.models as models
import torch.nn as nn
import cv2
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# preprocess/threshold images to pass into the network
def threshold_handwritten_image(filename):
    # Define the input and output folders
    input_folder = 'handwritten_digits'

    # Read the input image
    img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

    # Threshold the image using a threshold value of 128
    thresh_value = 128
    ret, thresh = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)

    inv_thresh = cv2.bitwise_not(thresh)

    return inv_thresh


# prediction function to test the network
def predict(model, data, targets):
    predictions = []
    with torch.no_grad():
        for i in range(len(data)):
            image = data[i]
            target = targets[i]
            output = model(image)
            _, predicted = torch.max(output, 1)
            pred = predicted.item()
            predictions.append(pred)
            print("target/predicted: {}/{}".format(target, pred))
            for j in range(output.size(1)):
                print(f"Output feature {j}: {output[0][j]:.2f}")
    return predictions


# used to predict the handwritten digits
def predict_handwritten_digits(model):
    handwritten_data = []
    handwritten_target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Define the folder containing the images
    folder_path = 'handwritten_digits'
    filenames = os.listdir(folder_path)
    sorted_filenames = sorted(filenames)
    # Loop through the folder and add each image to the list
    for filename in sorted_filenames:
        # Read the image and add it to the list
        img = threshold_handwritten_image(filename)
        tensor = transforms.ToTensor()(img)
        handwritten_data.append(tensor)

    predictions = predict(model, handwritten_data, handwritten_target)

    show_plot(handwritten_data, predictions, 10, 'Predication')


# plot the handwritten digit images
# show
def show_plot(mnist_dataset, example_targets, no_images, truth_or_pred):
    plt.figure()
    for i in range(no_images):
        # set the plot to be 2 by 3
        plt.subplot(4, 3, i+1)
        # set it to be a tight plot
        plt.tight_layout()
        # set a few parameters
        plt.imshow(mnist_dataset[i][0], cmap='gray', interpolation='none')
        plt.title("{}: {}".format(truth_or_pred, example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# main/driver method of this application
def main():
    batch_size_train = 64
    batch_size_test = 1000

    model = Net()
    network_state_dict = torch.load('./results/model.pth')
    model.load_state_dict(network_state_dict)
    train_loader, test_loader = get_data(batch_size_train, batch_size_test)
    test = enumerate(test_loader)
    batch_idx, (test_data, test_targets) = next(test)
    predictions = predict(model, test_data, test_targets)
    show_sample(test_data, predictions, 9, 'Predication')
    predict_handwritten_digits(model)


# start point of this program
if __name__ == '__main__':
    main()
