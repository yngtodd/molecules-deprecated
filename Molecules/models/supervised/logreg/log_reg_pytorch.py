from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import gzip, pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
# Step 0)
# Input Data

# Data set (3x1)

#with gzip.open("cont-mat.pkl.gz", 'rb') as f:
#    data, label = pickle.load(f)

#x_train, x_test, y_train, y_test = train_test_split(data, label, 0.2)
#x_train = Variable(torch.Tensor(x_train))
#x_test = Variable(torch.Tensor(x_test))
#y_train = Variable(torch.Tensor(y_train))
#y_test = Variable(torch.Tensor(y_test))

# Dataset Class
class ProteinData(Dataset):
    """
    X data is contact matrices (21,21)
    Y data is binary classification labels 0 or 1
    """
    
    def __init__(self, data, labels, transform=None):
        """
        Parameters:
        __________
       
        data : str
            Path to the data in pkl format.
        transform : callable, optional
            Optional transform to be applied on a sample.
        """
	#with gzip.open(data, 'rb') as f:
         #   self.data, self.labels = pickle.load(f)
     	self.data = np.load(data)
	self.labels = np.load(labels)
        self.transform = transform
	self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, label, 0.2)	
	# 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cont_mat = self.data[index]
        cont_label = self.labels[index]
        sample = {'cont_matrix': cont_mat, 'label': cont_label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Step 1)
# Build a model class that inherits from torch.nn.module

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(441,1) # One in and one out

    def forward(self, x):
        """
        In the foraward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        #print("x has shape {}".format(x.size()))
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


# Load data, add train data here
dataset = ProteinData(data="./cont-mat-data.npy", labels="./label.npy")
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
			  shuffle=True,
			  num_workers=1)

# Our Model
model = LogisticRegressionModel()

# Step 2)
# Construct loss and optimizer using pytorch API 

# Construct our loss function and an Optimizer. The call to model.parameters()
# In the SGD constructor will contain the learnable parametes of the two
# nn.Linear modules which are members of the Model.
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # lr = learning rate


# Step 3) forward, loss, backward, step

# Training loop
for epoch in range(500):
    for i, data in enumerate(train_loader):
        #Get the inputs
	inputs = data['cont_matrix']
	labels = data['label']
	arr = inputs.numpy()
	if(arr.shape != (32, 21, 21)):
	    print(i, arr.shape)
	    print(len(train_loader))
	    #assert(False)
	inputs = arr.reshape(32,441)
	#inputs, labels = Variable(torch.Tensor(inputs)), Variable(torch.Tensor(labels))
	#inn = inputs.resize_(32, 441)
        #inputs = inn.clone()
	#inputs = torch.Tensor(inputs)
	inputs = torch.from_numpy(inputs)
	labels = labels.float()
	inputs = inputs.float()
	inputs, labels = Variable(inputs), Variable(labels)

	
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.data[0])
	    #code through here for test
        # Zero gradients. perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Step 4) Testing

