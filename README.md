## Using an image database in pytorch and designing your own CNN
This repository includes a basic example of a pytorch image classification neural network on the MNIST dataset. This article stands as an explanation of why we took the approach that we did, as well as how, in hopes that it helps people understand how to design their own neural networks.

###### Required modules
```
pytorch
numpy
scikit-image
pandas
```

## Code overview

###### csvreader
The main thing that we wanted out of this network was to be able to see the images we were testing and training on. This meant that rather than downloading the MNIST database using pytorch's dataloader, we downloaded the images themselves and created a csv to complement them. Our csv included the paths to each image, as well as the ground truth label indicating which class (digit) that the image was.

This class reads in a csv. The csv is formatted with the first line as a header, and every subsequent line having an image path (relative from the location of this python file), followed by the ground truth classification of that image (0-9 for MNIST).

Below is a sample of the format of our csv:
```
img_path,class
./csvs/mnist/img1.jpg,5
./csvs/mnist/img2.jpg,0
./csvs/mnist/img3.jpg,4
./csvs/mnist/img4.jpg,1
```
Before we can use the class column in our network, we must one-hot encode it. This is an Nx1 matrix in which all values are 0, except the class's index, which is 1, and where N is the total number of classes. For 'img1', the one-hot encoding would be '[0,0,0,0,1,0,0,0,0,0]' (10x1 matrix, where index 5 is 1).

###### Layer
The Layer class represents a single convolutional layer of the network, followed by a batch normalization and ReLU activation. This class is not used for fully connected layers.

When using convolutional layers, you must ensure that the 'in_channels' of one layer equals the 'out_channels' of the previous layer.

###### OurNet
This is where we assemble the layers of our network. Rather than using a common and/or established layer architecture, we wanted to design our own. [Here](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba) are a few architectures if you want to know more.

Our layer architecture is as follows:
- 2 Convolutional Layers
- Max Pooling
- Another Convolutional Layer
- Average Pooling
- 1 Fully Connected Layer

Since we were using the classic MNIST dataset, our input into the network was a single 28x28 greyscale image. However, this becomes a 1x1x28x28 in practice. The first 1 is how many images we are inputing into the network at a time (see: "single"). The second is the number of color channels. Since our images are greyscale values, this is also 1 (If it were RGB, this number would be 3!). Finally, the 28x28 is rather obviously the resolution of our image.

The output
###### train

###### npmax

###### evaluate

###### main
