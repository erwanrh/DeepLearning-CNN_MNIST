#!/usr/bin/env python
# coding: utf-8

# CNN AND NN for handwritten digits recognition (MNIST)
# Using : Pytorch
# In[1]:

#conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch


#%% Imports
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm

# %% Training and validation functions : Feedforward with backpropagation and Gradient Descent

# Training consists of gradient steps over mini batch of data
def train(model, trainloader, loss, optimizer, epoch, num_epochs):
    # We enter train mode. It is important for layers such as dropout, batchnorm, ...
    model.train()
    
    loop = tqdm(trainloader)
    loop.set_description(f'Training Epoch [{epoch + 1}/{num_epochs}]')
    
    # We iterate over the mini batches of our data
    for inputs, targets in loop:
    
        # Erase any previously stored gradient
        optimizer.zero_grad()
        
        
        outputs = model(inputs) # Forwards stage (prediction with current weights)
        loss = criterion(outputs, targets) # loss evaluation
        
        loss.backward() # Back propagation (evaluate gradients) 
        
        
        # Making gradient step on the batch (this function takes care of the gradient step for us)
        optimizer.step() 
        
def validation(model, valloader, loss):
    # Do not compute gradient, since we do not need it for validation step
    with torch.no_grad():
        # We enter evaluation mode.
        model.eval()
        
        total = 0 # keep track of currently used samples
        running_loss = 0.0 # accumulated loss without averagind
        accuracy = 0.0 # accumulated accuracy without averagind (number of correct predictions)
        
        loop = tqdm(valloader) # This is for the progress bar
        loop.set_description('Validation in progress')
        
        
        # We again iterate over the batches of validation data. batch_size does not play any role here
        for inputs, targets in loop:
            # Run samples through our net
            outputs = model(inputs)

            # Total number of used samples
            total += inputs.shape[0]

            # Multiply loss by the batch size to erase averagind on the batch
            running_loss += inputs.shape[0] * loss(outputs, targets).item()
            
            # how many correct predictions
            accuracy += (outputs.argmax(dim=1) == targets).sum().item()
            
            # set nice progress meassage
            loop.set_postfix(val_loss=(running_loss / total), val_acc=(accuracy / total))
        return running_loss / total, accuracy / total


# We use again the MNIST dataset. This time we will use the official train/test split!



# %% MNIST DATA
# We download the train set
all_train = datasets.MNIST('data/',
                           download=True,
                           train=True,
                           transform=transforms.ToTensor())

# We split the whole train set in two parts:
# the one that we actually use for training
# and the one that we use for validation
batch_size = 32 # size of the mini batch
num_train = int(0.8 * len(all_train))

trainset, valset = torch.utils.data.random_split(all_train, [num_train, len(all_train) - num_train])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

#Downloading the test set
testset = datasets.MNIST('data/',
                         download=True,
                         train=False,
                         transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


# %% Net + training parameters
num_epochs = 2 # how many passes over the whole train data
input_size = 784 # flattened size of the image
hidden_sizes = [128, 64] # sizes of hidden layers
output_size = 10 # how many labels we have
lr = 0.001 # learning rate
momentum = 0.9 # momentum

#%% MODEL 1 : Multi layer perceptron -------------------

class SimpleFeedForward(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64],
                 output_size=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]), 
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        )
             
    def forward(self, x):
        x = x.reshape(-1, input_size)
        x = self.classifier(x)
        return x


# %% Training the first model : simple multilayer perceptron

# initializing our model/loss/optimizer
net = SimpleFeedForward(input_size, hidden_sizes, output_size) # Our neural net
criterion = nn.CrossEntropyLoss() # Loss function to be optimized
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum) # Optimization algorithm

# num_epochs indicates the number of passes over the data
for epoch in range(num_epochs):
    
    # makes one pass over the train data and updates weights
    train(net, trainloader, criterion, optimizer, epoch, num_epochs)

    # makes one pass over validation data and provides validation statistics
    val_loss, val_acc = validation(net, valloader, criterion)


# %% Validation 
test_loss, test_acc = validation(net, testloader, criterion)
print(f'Test accuracy: {test_acc} | Test loss: {test_loss}')



#%% MODEL 2 : Convolutional network  -------------------

class ConvNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=[1, 1], padding=2), #Convolution Layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(
            nn.Linear(14 * 14 * 8, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


#%% Training of model 2
net1 = ConvNet1()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net1.parameters(), lr=lr)

for epoch in range(num_epochs):
    
    # makes one pass over the train data and updates weights
    train(net1, trainloader, criterion, optimizer, epoch, num_epochs)

    # makes one pass over validation data and provides validation statistics
    val_loss, val_acc = validation(net1, valloader, criterion)


#%% Validation

test_loss1, test_acc1 = validation(net1, testloader, criterion)
print(f'Test accuracy: {test_acc1} | Test loss: {test_loss1}')

#Better results than the simple multilayer perceptron
#Takes a longer time to compute


#%% ConvNet architecture with Dropout layer


class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=[1, 1], padding=2), #Convolution Layer
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(
            nn.Linear(14 * 14 * 8, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


#%% Training of model 3
net2 = ConvNet2()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net2.parameters(), lr=lr)

for epoch in range(num_epochs):
    
    # makes one pass over the train data and updates weights
    train(net2, trainloader, criterion, optimizer, epoch, num_epochs)

    # makes one pass over validation data and provides validation statistics
    val_loss, val_acc = validation(net2, valloader, criterion)


# %% Validation of 3rd model 
test_loss2, test_acc2 = validation(net2, testloader, criterion)
print(f'Test accuracy: {test_acc2} | Test loss: {test_loss2}')






