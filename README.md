## A sample of using an image database in pytorch and designing your own convolutional neural network
This repository includes a basic example of a pytorch image classification neural network on the MNIST dataset. This article stands as an explanation of why we took the approach that we did, as well as how, in hopes that it helps people understand how to design their own neural networks.

The main thing that we wanted out of this network was to be able to see the images we were testing and training on. This meant that rather than downloading the MNIST database using pytorch's dataloader, we downloaded the images themselves and created a csv to complement them. Our csv included the paths to each image, as well as the ground truth label indicating which class (digit) that the image was. The code we provide in the `csvreader()` function handles loading of the csv document.
  
###### Required modules
```
pytorch
numpy
scikit-image
pandas
```

###### Getting started

