# Simple CNN example on the MNIST dataset
# Two conv layers, handwriting classification
import os
import random
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Hyperparameters adjustable to get optimal results
EPOCH = 5 # Total epochs for training
BATCH_SIZE = 50 # Batchsize
LR = 0.001 # Learning rate
NFILTER = [16,32] # Number of filters for each conv layer
KSIZE = [3,5] # kernel size for each layer,
STRIDE = [1,1] # Stride
PADDING = [1,2] # PADDING
POOLSIZE = [2,2] # Pooling size

# Download MNIST dataset from torchvision
Download_Minist = True
if os.path.exists('./mnist/'):
    Download_Minist = False # dataset exist, no need to download again
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # True for training and false for test dataset
    download=Download_Minist,  # True for download data, False for having downloaded
)

# Print one example
print(train_data.train_data.size())                 # (60000, 28, 28) 60000 images in total
print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
# plt.show()

# To speed up training, randomly select 10000 from total 60000 images for training
trainNum = train_data.train_data.size()[0] #60000
sampleInd = random.sample(range(0, trainNum), 10000)
train_x = torch.unsqueeze(train_data.train_data, dim=1).type(torch.FloatTensor)[sampleInd,:] # select 10000 images for training
                                                                                             # size [10000, 1, 28, 28]
train_x = train_x/255   # normalize images from [0, 255] to [0, 1.0]
train_y = train_data.train_labels[sampleInd]
# Using dataloader for mini-batch organization, each batch is of shape (BATCH_SIZE, 1, 28, 28)
train_set = TensorDataset(train_x, train_y)
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

# To speed up select first 1000 samples from testing dataset to do test
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[0:1000]/255
test_y = test_data.test_labels[0:1000]

def total_count(loader):
    totalClassCount = [0,0,0,0,0,0,0,0,0,0]

    for batch_id,(images,labels) in enumerate(loader):
        for label in labels:
            totalClassCount[int(label)] += 1
    return totalClassCount

classes = [0,1,2,3,4,5,6,7,8,9]
print("Digit class = ",classes)
totalCount = total_count(train_loader)

fig0, ax0 = plt.subplots()
ax0.barh(y=classes,width=totalCount)
ax0.set_xlabel('# Examples')
ax0.set_ylabel('# Digit Classes')
ax0.set_title('Train Set')
temp = train_loader.dataset[0][0].numpy()
temp = np.reshape(a=temp,newshape=(temp.shape[1],temp.shape[2]))
plt.imshow(temp)

# The function to calculate the output size of Conv layer
def calConvSize(lin, kernel, stride, padding=0, dilation=1):
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)
# The function to calculate the output size of pooling layer
def calPoolSize(lin, kernel, stride=None, padding=0, dilation=1):
    if stride is None:
        stride = kernel
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)

# class CNN(nn.Module):
#     # Two conv layers CNN network
#     def __init__(self, nfilter=[16,32], kernelsize=[5,5], stride=[1,1], padding=[2,2], poolsize=[2,2]):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
#             nn.Conv2d(
#                 in_channels=1,              # input channel number = 1
#                 out_channels=nfilter[0],    # filter number == output channel number
#                 kernel_size=kernelsize[0],  # filter size
#                 stride=stride[0],           # filter movement/step stride
#                 padding=padding[0],         # padding size
#             ),
#             nn.ReLU(),                      # activation function
#             nn.MaxPool2d(kernel_size=poolsize[0]),    # max pooling
#         )
#         Lout = calConvSize(lin=28, kernel=kernelsize[0], stride=stride[0], padding=padding[0]) # conv output size
#         Lout = calPoolSize(lin=Lout, kernel=poolsize[0]) # pooling output size
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(nfilter[0], nfilter[1], kernelsize[1], stride[1], padding[1]),
#             nn.ReLU(),
#             nn.MaxPool2d(poolsize[1]),
#         )
#         Lout = calConvSize(lin=Lout, kernel=kernelsize[1], stride=stride[1], padding=padding[1]) # conv output size
#         Lout = calPoolSize(lin=Lout, kernel=poolsize[1]) # pooling output
#         self.out = nn.Linear(nfilter[1] * Lout * Lout, 10)   # fully connected layer, output 10 classes

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)           # flatten the output of conv2 to fully connected layer
#         output = self.out(x)
#         return output    # return 10 scores

class LeNet5(nn.Module):
    # test LeNet5
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        
        self.fc1 = nn.Linear(1024,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.fc1Size = 0
        self.toKnowMaxPoolSize= False
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),1)
        
        if(self.toKnowMaxPoolSize == True):
            self.fc1Size = x.size()
            print(x.size())
            return
        #now lets reshape the matrix i.e. unrolling the matrix
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

n1 = LeNet5()
n1 = n1.cuda()

cnn = CNN(nfilter=NFILTER, kernelsize=KSIZE, stride=STRIDE, padding=PADDING, poolsize=POOLSIZE) # Define network
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_fun = nn.CrossEntropyLoss()


# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # Iterate for each minibatch

        output = cnn(b_x)              # cnn output
        loss = loss_fun(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| Train loss: %.4f' % loss.data.numpy(), '| Test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
# plot these 10 images
fig, axes = plt.subplots(2,5, figsize=(10,4), tight_layout=True)
axes = axes.flat
for ii in range(len(axes)):
    ax = axes[ii]
    ax.imshow(test_x[:10].numpy()[ii,0,:], cmap='gray')
    ax.set_title('Prediction %i' % pred_y[ii])