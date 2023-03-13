"""
Generates a bunch of 32x32 pixel Red, Green, or Blue thumbnails of different shades.
Used for both training and then later validating an image classifier neural network.

This code is as simple as possible for learning purposes.

Kris Pritchard / @krp - 2023
"""
import os
import random
from PIL import Image

# Define the root directory for the datasets
root_dir = './data'

# Define the number of images in the training and validation sets
num_train = 16  # Realistically you'd have a much larger training set, but this is for experimental purposes.
num_val = 1024  # Your validation set would typically be 10% of your training set. I wanted far more here to check the accuracy of the classifier.

# Define the shades of red, green, and blue to use in the images.
# These shades are used to train the network.
red_shades = [(255, 0, 0), (200, 0, 0), (150, 0, 0), (100, 0, 0), (50, 0, 0)]
green_shades = [(0, 255, 0), (0, 200, 0), (0, 150, 0), (0, 100, 0), (0, 50, 0)]
blue_shades = [(0, 0, 255), (0, 0, 200), (0, 0, 150), (0, 0, 100), (0, 0, 50)]

# Create the training set
train_dir = os.path.join(root_dir, 'training')
os.makedirs(train_dir, exist_ok=True)

for i in range(num_train):
    # Randomly select a shade of red, green, or blue. With a small dataset for training this will lead to some bias.
    # Observing how it affects the detection quality can be interesting though. (e.g. if you train it on mostly reds)
    color = random.choice(['red', 'green', 'blue'])
    if color == 'red':
        shade = random.choice(red_shades)
        #shade = (random.randint(1, 255), 0, 0)  # Uncomment if you want to train it on more randomized colors.
    elif color == 'green':
        shade = random.choice(green_shades)
        #shade = (0, random.randint(1, 255), 0)
    else:
        shade = random.choice(blue_shades)
        #shade = (0, 0, random.randint(1, 255))

    # Create the image with the chosen shade and save it as color_number.png. e.g. red_13.png
    image = Image.new('RGB', (32, 32), shade)
    filename = f'{color}_{i}.png'
    filepath = os.path.join(train_dir, filename)
    image.save(filepath)

print('Finished generating training data. Generating validation data.')

# Create the validation set, ensure that colors can be any shade of red, green, or blue.
val_dir = os.path.join(root_dir, 'validation')
os.makedirs(val_dir, exist_ok=True)
for i in range(num_val):
    # Randomly select a shade of red, green, or blue
    color = random.choice(['red', 'green', 'blue'])
    if color == 'red':
        #shade = random.choice(red_shades)
        shade = (random.randint(1, 255), 0, 0)
    elif color == 'green':
        #shade = random.choice(green_shades)
        shade = (0, random.randint(1, 255), 0)
    else:
        #shade = random.choice(blue_shades)
        shade = (0, 0, random.randint(1, 255))

    # Create the image with the chosen shade and save it
    image = Image.new('RGB', (32, 32), shade)
    filename = f'{color}_{i}.png'
    filepath = os.path.join(val_dir, filename)
    image.save(filepath)

print('Finished generating validation data. Ready to be used. It can help to verify the colors with an image editing app.')