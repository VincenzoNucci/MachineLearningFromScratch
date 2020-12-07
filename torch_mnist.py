import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_shape = 28*28
num_classes = 10
epochs = 100
batch_size = 128
batches = int(np.ceil(60000/batch_size))
layers = 3

transform = transforms.Compose([
  transforms.ToTensor()
])

train_set = MNIST('dataset',train=True, download=True, transform=transform)
test_set = MNIST('dataset',train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

# X_train = train_set.data
# y_train = train_set.targets
# X_test = test_set.data
# y_test = test_set.targets
# X_train = X_train.reshape((-1,784)) / 255.0
# X_test = X_test.reshape((-1,784)) / 255.0



weights_layer = {e:{} for e in range(epochs)}
for e in range(epochs):
    for b in range(batches):
        weights_layer[e][b] = []

# initial = {e:[] for e in range(layers)}
initial = {}

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(input_shape,64,bias=False)
      self.dropout = nn.Dropout(p=0.5)
      self.fc2 = nn.Linear(64,64,bias=False)
      self.fc3 = nn.Linear(64,num_classes,bias=False)
      #initial[0] = self.fc1.weight
      initial[0] = self.fc2.weight
      #initial[2] = self.fc3.weight

    # x represents our data
    def forward(self, x, epoch, batch, training=None):
      # Pass data through conv1
      x = self.fc1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.dropout(x)
      if training:
        bs = np.zeros((batch_size,*self.fc2.weight.shape))
        for a, activation in enumerate(x):
          with torch.no_grad():
            bs[a] = self.fc2.weight*activation
        weights_layer[epoch][batch].append(bs)
      x = self.fc2(x)

      x = F.relu(x)

      x = self.fc3(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output

model = Net()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
loss_list = np.zeros((epochs,))
accuracy_list = np.zeros((epochs,))

import tqdm

for epoch in range(epochs):
  running_loss = 0.0
  running_accuracy = 0.0
  for i, (batch_x, batch_y) in enumerate(train_loader):
    inputs = batch_x.reshape((-1,28*28)).to(device)
    labels = batch_y.to(device)
    # Forward pass
    y_pred = model(inputs,epoch,i,training=True)
    loss = loss_fn(y_pred, labels)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    
    # Accuracy
    with torch.no_grad():
      correct = 0
      total = 0
      for j, (test_x, test_y) in enumerate(test_loader):
        inputs = test_x.reshape((-1,28*28))
        labels = test_y.to(device)

        y_pred = model(inputs,epoch,i)
        _, predicted = torch.max(y_pred.data,dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        running_accuracy += correct.item() / total

    print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{batches}], train_Loss: {running_loss:.4f}, train_Accuracy: {running_accuracy/batches:.4f}')
  loss_list[epoch] = running_loss/batches
  accuracy_list[epoch] = running_accuracy/batches

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

ax1.plot(accuracy_list.mean())
ax1.set_ylabel("validation accuracy")
ax2.plot(loss_list.mean())
ax2.set_ylabel("validation loss")
ax2.set_xlabel("epochs");
plt.show()

# import os
# import pickle
# model = my_nn
# epochs = 10
# batch_size = 128
# batches = int(np.ceil(len([1])/batch_size))
# mlayers = len(model.layers)

# weights = np.zeros((epochs,batches,mlayers),dtype=object)

# for root, dirnames, filenames in os.walk('time_series_batch'):
#   for epoch, filename in enumerate(sorted(filenames)):
#     with open(os.path.join(root,filename),'rb') as f:
#       data = pickle.load(f)
#       for layer in range(mlayers):
#         for batch in range(batches):
#           if len(data[layer]) > 0:
#             weights[epoch,batch,layer] = data[layer][batch]