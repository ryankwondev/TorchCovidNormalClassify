# import torch things
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


# load model.pth and get result with image.png
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # 수정된 부분

        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()  # 수정된 부분

    def forward(self, x):
        # x = x.reshape(x.size(0), -1)  # Flatten the tensor

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# load model.pth
model = NeuralNetwork()
state_dict = torch.load('model0.pth')
model.load_state_dict(state_dict)

# get result with image.jpg
image = cv2.imread('image.jpg')
image = cv2.resize(image, (128, 128))
image = image / 255.0  # Normalize the image
image = image.astype(np.float32)  # Ensure the image is float32
image = torch.from_numpy(image)
image = image.unsqueeze(0)
image = image.permute(0, 3, 1, 2)
print(image.shape)



result = model(image)
print(result)

import torch.nn.functional as F

probabilities = F.softmax(result, dim=1)
print(probabilities)
