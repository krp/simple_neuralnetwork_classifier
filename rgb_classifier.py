"""
Simple neural network for classifying if a picture is red, green, or blue.

This code is both simple and verbose for learning purposes.

Usage:
Generating images:
- Run the generate_trainingset.py file to generate a bunch of 32x32 RGB images.

Training the model:
- Specify the NUM_EPOCHS you want below. Experiment with different values to see how it changes prediction accuracy.
- Running the command (without the $) will train on images in the ./data/training directory.
$ python rgb_classifier.py train

Validating the accuracy of the model:
- Running the command (also without the $) will open the .pth model and check it against the ./data/validation images.
$ python rgb_classifier.py validate

Testing it against a single image:
- Give it an image and get it to tell you whether it thinks the image is red, green, or blue.
$ python rgb_classifier.py somefile.png

Once you understand how this code works, modify it to try and solve other classification problems.
e.g. Hotdog or Not Hotdog.

Try running it on some of the mixed colors in the ./individual_test_colors directory too.
e.g.
$ python rgb_classifier.py pinkbutmorered_1.png

Kris Pritchard / @krp - 2023
"""
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Uncomment these to experiment with different amounts of training and see how the prediction accuracy changes.
#NUM_EPOCHS = 2  # 62% accuracy with 14 training pics, 8 test pics, and 2 epochs
#NUM_EPOCHS = 20  # 81% accuracy with 14 training pics, 8 test pics, and 20 epochs
NUM_EPOCHS = 100  # 100% accuracy with 14 training pics, 8 test pics, and 20 epochs
PATH = f'./rgb_predictor_{NUM_EPOCHS}_epochs.pth'
TRAINING_DIR = './data/training'
VALIDATION_DIR = './data/validation'
NUM_TRAINING_FILES = 8

# The colors we care about. This tuple can be changed to handle as many cases as you like.
classes = ('red', 'green', 'blue')

# Simple helper function used to display an image.
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class SingleFileDataset(Dataset):
    """Used for a single file. Separate class for demonstration purposes."""
    def __init__(self, filename):
        self.filepath = filename
        # Could move this elsewhere but keeping it for demonstration purposes.
        self.transforms = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32 if images are different sizes.
            transforms.ToTensor(),  # Transform them into tensors.
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Naively normalize them. 
        ])
    
    def __getitem__(self, index):
        # Could also make this a one-liner but keeping it like this for teaching.
        image = Image.open(self.filepath)
        image = self.transforms(image)
        return image
    
    def __len__(self):
        return 1


# Dataset used for training.
class ColorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # You can normalize by the actual mean and standard deviation if you like, but it's not necessary in this simple example.
        self.transforms = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32 if images are different sizes.
            transforms.ToTensor(),  # Transform them into tensors.
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Naively normalize them. 
        ])

        self.filenames = os.listdir(self.root_dir)
        self.labels = []
        for filename in self.filenames:
            color = filename.split('_')[0]
            label = {'red': 0, 'green': 1, 'blue': 2}[color]
            self.labels.append(label)
        
        # Adds more 'fake' data to the dataset if enough aren't provided. (just adds the last image repeatedly)
        while len(self.filenames) % NUM_TRAINING_FILES != 0:
            self.filenames.append(self.filenames[-1])
            self.labels.append(self.labels[-1])

        print(f'Loaded {len(self.filenames)} files.')

    def __len__(self):  # Used for debugging. e.g. len(dataset)
        return len(self.filenames)

    def __getitem__(self, idx):  # Used for accessing items in the dataset. e.g. dataset[42]
        filename = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(filename).convert('RGB')
        image = self.transforms(image)
        label = self.labels[idx]
        return image, label


class Net(nn.Module):
    def __init__(self):
        # Magically chosen Convolutional Neural Network (CNN) architecture follows.
        # I found these values carved into a cliff on a mountainside and they appear to work..
        super(Net, self).__init__()
        # 3 input image channel (RGB), 6 output channels, 5x5 square convolution
        # kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = F.relu(self.conv1(x))  # Test if you want to try without pooling.
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Used to train the neural net
def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = ColorDataset(TRAINING_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    net = Net()
    net.to(device)  # send to GPU if available
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)  # Try a different optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            print(f'Training Loop: {i}')
            # Get the inputs, data is a list of [inputs, labels]
            #inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)  # Send it to GPU if available

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini batches
                print(f'[{epoch + 1}], [{i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    print(f'Finished Training. Saving to {PATH}')

    # Save
    torch.save(net.state_dict(), PATH)

def validate_accuracy():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_dataset = ColorDataset(VALIDATION_DIR)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    net = Net()
    net.load_state_dict(torch.load(PATH))
    print(f'Opened model {PATH} for validation testing.')
    net.to(device)

    print(f'Test Dataset length: {len(test_dataset)}')
    print(f'Test Dataset filenames: {test_dataset.filenames}')

    """ Loopity loop. Uncomment this if you want to see how it behaves on each individual validation file.
    #dataiter = iter(test_loader)
    #data = next(dataiter)
    with torch.no_grad():
        for data in test_loader:
            print('======')
            print(data)
            print('======')

            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            print('Labels[0]:')
            print(labels[0])
            print('Predicted[0]')
            print(predicted[0])
            print(f'Actual Label: {classes[labels[0]]}')
            print(f'Predicted: {classes[predicted[0]]}')
            print('===')

    # Uncomment this if you want to see some of the images.
    #print('Data: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(2)))
    #imshow(torchvision.utils.make_grid(images))
    """


    """ Accuracy. Predicts all the validation data values then compares them with their actual colors."""
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            # NOTE: Remember to send the data to the GPU if you get: "RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor"
            #images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network: {100 * correct // total}%')


def check_image_color_prediction(filename):
    """Takes a filename and runs a neural network on it."""
    print(f'Predicting the color of: {filename}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classify_dataset = SingleFileDataset(filename)
    classify_loader = torch.utils.data.DataLoader(classify_dataset, batch_size=1)

    net = Net()
    net.load_state_dict(torch.load(PATH))
    print(f'Opened model {PATH} for validation testing.')
    net.to(device)

    #dataiter = iter(classify_loader)  # Create an iterator.
    #images = next(dataiter)  # Grab the first item
    with torch.no_grad():
        #images, labels = data[0].to(device), data[1].to(device)
        for data in classify_loader:
            inputs = data.to(device)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, dim=1)
            print(f'{filename} predicted to be {classes[predicted]} - How\'d I do?')



if __name__ == '__main__':
    # Grab an action from the command line.
    program_name = sys.argv[0].rstrip('.py')
    if len(sys.argv) != 2:
        print(f'Usage: {program_name} <train | validate | filename>')
        print(f'e.g. {program_name} train will train the neural net, validate will validate it, and filename will check against a file.')
        sys.exit()

    action = sys.argv[1]
    if action == 'train':
        print('Training..')
        train()
    elif action == 'validate':
        print('Validating accuracy..')
        validate_accuracy()
    else:
        filename = action
        print(f'Predicting color of filename {filename}')
        check_image_color_prediction(filename)
