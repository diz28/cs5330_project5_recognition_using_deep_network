
# Di Zhang
# Mar 31, 2023
# CS5330 - Computer Vision

import cv2
import torch
from torchvision import transforms
from main import Net


# the main/driver class of the application
def main():
    model = Net()
    model.eval()
    network_state_dict = torch.load('./results/model.pth')
    model.load_state_dict(network_state_dict)

    # Set up a video capture object to get frames from the camera
    cap = cv2.VideoCapture(0)

    freeze_frame = False

    while True:
        # Get the next frame from the camera
        ret, frame = cap.read()

        resized_img = cv2.resize(frame, (28, 28), interpolation=cv2.INTER_AREA)

        #frame = resized_img

        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        # Threshold the image using a threshold value of 128
        thresh_value = 65
        ret, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)

        inv_thresh = cv2.bitwise_not(thresh)

        tensor = transforms.ToTensor()(inv_thresh)

        with torch.no_grad():
            output = model(tensor)
            _, predicted = torch.max(output, 1)
            pred = predicted.item()
            score = torch.max(output)

        # Draw the predicted digit and score on the frame
        cv2.putText(frame, f'Predicted digit: {pred} ({score:.2f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame with the predicted digit
        cv2.imshow('frame', frame)

        # check for the 'k' key to toggle the freeze flag
        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            freeze_frame = not freeze_frame

        # check for the 'q' key to exit
        if key == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()


# start point of the program
if __name__ == '__main__':
    main()