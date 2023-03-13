# Simple RGB Classifier using a Convolutional Neural Network

![training data thumbnails of red green and blue colors](https://raw.githubusercontent.com/krp/simple_neuralnetwork_classifier/main/colors.png)

Example of a super simple neural network that's trained on only 16 shades of  red, green, or blue, (see the ./data/training directory) and is able to identify what the primary color is in any other color it's given (see the ./training/validation and ./individual_test_color directories).

Play around with the code, delete training data, regenerate it with new colors, experiment with different training sizes, neural network architectures, different optimizers, and a different number of epochs to see how you can get its accuracy to change.

It's a kind of simple problem as we have better ways of getting the dominant color in an image, but this can be extended to use different datasets such as CIFAR-10 or Fashion-MNIST.

### Installation

Just run

```shell
git clone https://github.com/krp/simple_neuralnetwork_classifier
cd simple_neuralnetwork_classifier
pip3 install Pillow numpy matplotlib torch torchvision torchaudio # or use a virtualenv
```

If you've got a card that support CUDA (NVIDIA card) then you can also install that by adding `--extra-index-url https://download.pytorch.org/whl/cu117` to the end of the `pip3 install` command above. This network is tiny though so it's not going to make much difference on the CPU vs GPU.


### Training

Make sure you've got images in the ./data/training directory (use generate_trainingset.py if you don't) then run:

```shell
python rgb_classifier.py train
```

It'll create a `.pth` file which contains the model. This is then used to classify the dominant color of an image.

![picture of other colors it can classify](https://raw.githubusercontent.com/krp/simple_neuralnetwork_classifier/main/extracolors.png)

![image of the command line output of checking it against a color not in the training set](https://user-images.githubusercontent.com/2504972/224700993-e3f65528-9b84-404f-8a80-7cad0c253bd7.png)


### Validation

Run

```shell
python rgb_classifier.py validate
```

It'll check the model's predictions against the 1024 images in the `./data/validation` directory and tell you how accurate it was. Experiment with the `NUM_EPOCHS` setting and see how few epochs it takes to get it to 100% accuracy.

### Standalone Images

Run it on any of the files in the `./individual_test_color` directory.
```shell
python rgb_classifier.py individual_test_color/bluedabadee.png
```

See what kind of prediction accuracy you get by also verifying the colors using the eye-dropper tool in an image editor. See what kind of results you get against the full white, grey, and black images when the network has no choice but to give one of three answers.

### Experimentation & Learning

Play around with the different settings in the `rgb_classifier.py` file.

© Kris Pritchard / @krp - 2023
