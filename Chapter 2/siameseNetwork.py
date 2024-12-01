import argparse
import torch
import torchvision.datasets as dsets
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from sklearn.manifold import TSNE

# Dataset class
class Dataset(object):
    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0).float() / 255.0
        self.x1 = torch.from_numpy(x1).float() / 255.0
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return (self.x0[index].unsqueeze(0),  # Add channel dimension
                self.x1[index].unsqueeze(0),  # Add channel dimension
                self.label[index])

    def __len__(self):
        return self.size

    def create_pairs(self, data, digit_indices):
        x0_data = []
        x1_data = []
        label = []
        n = min([len(digit_indices[d]) for d in range(10)]) - 1
        for d in range(10):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                x0_data.append(data[z1] / 255.)
                x1_data.append(data[z2] / 255.)
                label.append(1)
                inc = random.randrange(1, 10)
                dn = (d + inc) % 10
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                x0_data.append(data[z1] / 255.)
                x1_data.append(data[z2] / 255.)
                label.append(0)

        x0_data = np.array(x0_data, dtype=np.float32).reshape([-1, 1, 28, 28])
        x1_data = np.array(x1_data, dtype=np.float32).reshape([-1, 1, 28, 28])
        label = np.array(label, dtype=np.int32)
        return x0_data, x1_data, label

# Contrastive loss function
def contrastive_loss_function(x0, x1, y, margin=1.0):
    diff = x0 - x1
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)
    mdist = margin - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    loss = torch.sum(loss) / (2.0 * x0.size()[0])
    return loss

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.Linear(10, 2)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Data preparation
batchsize = 128
train = dsets.MNIST(root='../data/', train=True, download=True)
test = dsets.MNIST(root='../data/', train=False, transform=transforms.Compose([transforms.ToTensor()]))

indices = np.random.choice(len(train.targets.numpy()), 2000, replace=False)
indices_test = np.random.choice(len(test.targets.numpy()), 100, replace=False)

train_iter = Dataset(train.data.numpy()[indices].astype(np.float32), train.data.numpy()[indices], train.targets.numpy()[indices])
test_iter = Dataset(test.data.numpy()[indices_test].astype(np.float32), test.data.numpy()[indices_test], test.targets.numpy()[indices_test])

train_loader = torch.utils.data.DataLoader(train_iter, batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_iter, batch_size=batchsize, shuffle=True)

# Model, optimizer, and loss
model = SiameseNetwork()
learning_rate = 0.001
momentum = 0.9
criterion = contrastive_loss_function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training loop
train_loss = []
epochs = 100
for epoch in range(epochs):
    for batch_idx, (x0, x1, labels) in enumerate(train_loader):
        labels = labels.float()
        x0, x1, labels = x0.float(), x1.float(), labels.float()
        output1, output2 = model(x0, x1)
        loss = criterion(output1, output2, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}')

# Plot loss function
def plot_loss(train_loss):
    plt.plot(train_loss, label="train loss")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

plot_loss(train_loss)
plt.savefig("train_loss.png")

# Testing and dimensionality reduction
def test_model(model):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for x0, x1, label in test_loader:
            x0, x1 = x0.float(), x1.float()
            output1, _ = model(x0, x1)
            embeddings.append(output1.numpy())
            labels.append(label.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(embeddings)
    return embeddings, labels

# Embedding visualization
def plot_mnist(numpy_all, numpy_labels, name="./embeddings_plot.png"):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    for i in range(10):
        f = numpy_all[np.where(numpy_labels == i)]
        plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.title("MNIST Test Set Embeddings")
    plt.savefig(name)

def testing_plots(model):
    numpy_all, numpy_labels = test_model(model)
    plot_mnist(numpy_all, numpy_labels)
    plt.savefig("embeddings_plot.png")

testing_plots(model)
