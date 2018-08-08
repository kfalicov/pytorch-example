import os
import pandas
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from numpy import array
from skimage import io
from random import randint

class csvreader():
    """This class reads in a csv. The csv is formatted with the first line
    as a header, and every subsequent line having an image path (relative 
    from the location of this python file), followed by the ground truth
    classification of that image (0-9 for MNIST)
    """
    def __init__(self, start_index, num_images):
        self.csv_root = './csvs'
        self.csv_filename = 'MNIST_csv.csv'
        self.csv_path = os.path.join(self.csv_root, self.csv_filename)
        if start_index + num_images > 10000:
            print("warning: exceeds csv size!")
        self.csv_data = pandas.read_csv(self.csv_path, skiprows=range(1,start_index), nrows=num_images)
    def __len__(self):
        return len(self.csv_data)
    # access image at idx in csv
    def __getitem__(self, idx):
        csvline = self.csv_data.iloc[idx]
        csvclass = csvline.loc['class']
        csvimgpath = csvline.loc['img_path']
        onehot = np.zeros(10)
        onehot[csvclass] = 1.
        image = io.imread(csvimgpath)
        returndict = {'image':np.array(image), 'class':onehot}
        return returndict

class Layer(nn.Module):
    """Represents a single layer of the network, with input and output channels 
    that extract features from input data. Each layer performs
    a convolution with kernel size 5, a batch normalization, and 
    a rectified linear unit (ReLU)
    """
    def __init__(self, in_channels, out_channels):
        super(Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output
   
class OurNet(nn.Module):
    """The network is where our particular arrangement of 'Layer()' objects go.
    OurNet uses two convolution layers, followed by a max pooling layer, another
    convolution layer, an average pooling layer, and a fully connected layer.
    """
    def __init__(self, num_classes=10):
        super(OurNet, self).__init__()
        self.layer1 = Layer(in_channels=1, out_channels=16)
        self.layer2 = Layer(in_channels=16, out_channels=16)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.layer3 = Layer(in_channels=16, out_channels=32)

        # (28)      (Image size)
        # -(4) = 24 (Kernel size for layer1 minus 1)
        # -(4) = 20 (Kernel size for layer2 minus 1)
        # /(2) = 10 (Kernel size for pooling layer)
        # -(4) = 6  (Kernel size for layer3 minus 1)
        # result: 6, the "magic number" used below
        self.avgpool = nn.AvgPool2d(kernel_size=6)

        self.net = nn.Sequential(self.layer1, self.layer2, self.pool1,
                                 self.layer3, self.avgpool)
        
        # in_features must be the same as the out_channel from the last used Layer
        self.fc = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,32) #must be same as last layer's out_channels
        output = self.fc(output) #make fully connected at end
        return output

def train(model):
    """Trains the model using the first 8000 images out of our 10000 image dataset
    """
    print("Training is starting now:")
    csvdata = csvreader(0, 8000) # read in the first 8000 lines of the csv
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # stochastic gradient descent
    loss_function = nn.CrossEntropyLoss() # combination of softmax and loss function

    num_epochs = 1
    for epoch in range(num_epochs):
        # At the moment, we don't do augmentation, so this section is empty
        #
        running_loss = 0
        for idx in range(len(csvdata)): # look at one line of csv at a time
            optimizer.zero_grad()
            
            # the unsqueeze_ calls found here are to ensure that the
            # dimensionality of everything is consistent by adding
            # an extra dimension to the matrix
            image_to_guess = torch.from_numpy(csvdata[idx]['image']).unsqueeze_(0).float()
            image_to_guess = image_to_guess.unsqueeze_(0)

            ground_truth_class = torch.from_numpy(csvdata[idx]['class']).long().unsqueeze_(0)

            # Variable objects are special in that they remember not just
            # a value, but the calculations which led to that value.
            image_to_guess, ground_truth_class = Variable(image_to_guess), Variable(ground_truth_class)
            guess = model(image_to_guess)
            loss = loss_function(guess, torch.max(ground_truth_class,1)[1])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % 1000 == 999:
                print('Epoch: %d, batch: %d, loss: %.5f' % (epoch+1, idx+1, running_loss/1000))
                running_loss = 0.0

def npmax(l):
    """A helper function that finds the index of the max value in a given list,
        and returns a tuple containing it as well as the value at that index.
    """
    max_idx = np.argmax(l)
    max_val = l[max_idx]
    return (max_idx, max_val)

def evaluate(model):
    """Gives the user feedback on the accuracy of the model
    """
    print("Evaluation is starting now:")
    csvdata = csvreader(8000, 2000) # starting with index 8000, reads the next 2000 lines of the csv
    
    # running count of correct image predictions
    correct = 0
    total = 0

    # to keep track of detection accuracy per-number
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    with torch.no_grad():
        for idx in range(len(csvdata)):
            image_to_guess = torch.from_numpy(csvdata[idx]['image']).unsqueeze_(0).float()
            image_to_guess = image_to_guess.unsqueeze_(0)
            ground_truth_class = npmax(csvdata[idx]['class'])[0]
            guess = model(image_to_guess).numpy() # converts the tensor guess to a numpy array
            
            total += 1
            correct += (npmax(guess[0])[0] == ground_truth_class)


            # Randomly chooses images to print so users can verify the network is functional
            if randint(0,250)==5:
                print("Randomly chosen image: img%d.jpg (g_t:%d), guess: %d" % (idx+8000, ground_truth_class, npmax(guess[0])[0]))

            class_total[ground_truth_class] += 1
            class_correct[ground_truth_class] += (npmax(guess[0])[0] == ground_truth_class)

            if idx % 500 == 499:
                print('Evaluating on image: %d, running accuracy: %d%%' % (idx+1, 100*correct/total))
    for i in range(10):
        print("Accuracy of recognizing %d: %d%%" % (i, 100*class_correct[i]/class_total[i]))
    print("Total accuracy on %d images: %d%%" % (len(csvdata), 100*correct/total))

def main():
    # To initialize a new model from scratch use:
    model = OurNet(num_classes=10)
    # otherwise, uncomment this line and pass in your existing model:
    #model = torch.load('model.pt')
    train(model)
    evaluate(model)
    print('Saving to main_results.pt')
    torch.save(model, 'main_results.pt')

if __name__ == '__main__':
    main()
