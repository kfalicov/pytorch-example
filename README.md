## A sample of using an image database in pytorch
This repository is an initial test of a pytorch image classifier on the MNIST dataset.

A few changes we made over the pytorch examples:
  We wanted to have a visual representation of all of the images in a folder. In order to access images from a folder for our network, we decided on a comma-separated values list, which would be a nice way of looking up images by index, and storing the ground truth values of each image so that both could be accessed at once. The code we provide in the `csvreader()` function handles loading of the csv document.
  
  
###### Required modules
```pytorch
numpy
scikit-image
pandas```
