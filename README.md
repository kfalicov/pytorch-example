## Using an image database in pytorch and designing your own CNN
by Paul Hughes and Kyle Falicov

This repository includes a basic example of a pytorch image classification neural network on the MNIST dataset. This article stands as an explanation of why we took the approach that we did, as well as how, in hopes that it helps people understand how to design their own neural networks. More in-depth comments can be found in `main.py`.

###### Required modules
```
pytorch
numpy
scikit-image
pandas
```

## Using our example

If you would like to try out our example, download the repository and extract `mnist.rar` into the `data` folder. Then run `main.py`.

## Code overview

###### csvreader
The main thing that we wanted out of this network was to be able to see the images we were testing and training on. This meant that rather than downloading the MNIST database using pytorch's dataloader, we downloaded the images themselves and created a csv to complement them. Our csv included the paths to each image, as well as the ground truth label indicating which class (digit) that the image was.

This class reads in a csv. The csv is formatted with the first line as a header, and every subsequent line having an image path (relative from the location of this python file), followed by the ground truth classification of that image (0-9 for MNIST).

Below is a sample of the format of our csv:
```
img_path,class
./data/mnist/img1.jpg,5
./data/mnist/img2.jpg,0
./data/mnist/img3.jpg,4
./data/mnist/img4.jpg,1
```
Before we can use the class column in our network, we must one-hot encode it. This is an Nx1 matrix in which all values are 0, except the class's index, which is 1, and where N is the total number of classes. For `img1`, the one-hot encoding would be [0,0,0,0,1,0,0,0,0,0] (10x1 matrix, where index 5 is 1). This represents the ideal 0% confidence for all incorrect classes, and 100% confidence for the correct class. To avoid type errors, make sure that these numbers are floats!

###### Layer
The Layer class represents a single convolutional layer of the network, followed by a batch normalization and ReLU activation. This class is not used for fully connected layers.

When using convolutional layers, you must ensure that the `in_channels` of one layer equals the `out_channels` of the previous layer.

###### OurNet
This is where we assemble the layers of our network. Rather than using a common and/or established layer architecture, we wanted to design our own. [Here](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba) are a few architectures if you want to know more.

Our layer architecture is as follows:
- 2 Convolutional Layers
- Max Pooling
- Another Convolutional Layer
- Average Pooling
- 1 Fully Connected Layer

Since we were using the classic MNIST dataset, our input into the network was a single 28x28 greyscale image. However, this becomes a 1x1x28x28 in practice. The first 1 is how many images we are inputing into the network at a time (see: "single"). The second is the number of color channels. Since our images are greyscale values, this is also 1 (If it were RGB, this number would be 3!). Finally, the 28x28 is rather obviously the resolution of our image.

The output of the network is a 10x1 matrix with confidence values for each class, represented as a percentage [0-1]. The highest confidence value in the output matrix is the network's guess as to which class it believes the input image belongs to.

###### train
The training class combines all of the other classes in order to make our network actually learn! First, it inputs an image into the network, and compares the output of the network to the `ground_truth` (the ideal output). Then the `loss` is calculated and the the weights of the network are tweaked using `backward()` (see: back propogation). The process then repeats, slowly improving the network towards a local minimum.

If this part confuses you, think of the network as a curve with multiple hills and valleys. Since our starting weights are random, we start on a random point along the curve. Next, we look and see which direction is downhill (see: gradient descent). We then take a small step (see: learning rate) in the downhill direction, getting closer to the nearest valley. If we repeat this process, we will eventually be right at the bottom, in the center of a valley. Ideally our goal is to be in the lowest valley along the whole curve, but the closest one will have to do. In reality, the math is a bit more complex than what was just described, but you get the idea.

###### evaluate
The evaluate class gives the user feedback on the accuracy of the model. After training our network, we used 2000 images from the MNIST dataset that our network had not seen before. It then inputs each image into the network and compares the network's output to the `ground_truth`. It tallys up all correct guesses, and divides by the total number of test images to get the accuracy of our network. It also does this on a per class basis. In our case, the digit 8 was the least accurate. This makes sense to us, since an 8 has similar visual characteristics of a 0, and also of a 3, 6, and 9, so it is understandable that the network may get confused.
