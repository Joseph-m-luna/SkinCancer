import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class SimpleNet(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_size = 32):
        super(SimpleNet, self).__init__()
        self.input = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding="same")
        self.input_bn = nn.BatchNorm2d(32)

        #define hidden layers
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same")
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding="same")
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same")
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding="same")
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding="same")
        self.bn6 = nn.BatchNorm2d(8)
        self.conv7 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding="same")
        self.bn7 = nn.BatchNorm2d(8)
        self.conv8 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding="same")
        self.bn8 = nn.BatchNorm2d(4)

        # define output classifier
        self.linear1 = nn.Linear(4096, 512)
        self.bn_l1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn_l2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)
        self.bn_l3 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128, 32)
        self.bn_l4 = nn.BatchNorm1d(32)

        self.output = nn.Linear(32, num_classes)

        # define activation/flatten layers
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4x4 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout20 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.input(x)
        # print("x shape: ", x.shape)
        x = self.input_bn(x)
        x = self.relu(x)
        # x = self.dropout40(x)
        #x = self.pool2x2(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool2x2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool2x2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.pool2x2(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)

        embedding = self.flatten(x)
        x = self.linear1(embedding)
        x = self.bn_l1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.bn_l2(x)
        x = self.relu(x)

        x = self.linear3(x)
        x = self.bn_l3(x)
        x = self.relu(x)

        x = self.linear4(x)
        x = self.bn_l4(x)
        x = self.relu(x)

        x = self.output(x)

        if self.training:
            return x, embedding.detach()
        else:
            return x

class Adversary(nn.Module):
    def __init__(self, input_size=256):
        super(Adversary, self).__init__()
        self.input = nn.Linear(input_size, 64)
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.output(x)
        return x
