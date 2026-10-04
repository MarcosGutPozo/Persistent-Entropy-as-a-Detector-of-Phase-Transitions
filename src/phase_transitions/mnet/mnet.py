import torch
import torch.nn as nn
import torch.nn.functional as F


class MNet(nn.Module):

    def __init__(
        self,
        in_channels=1,   # Input channels (1 for grayscale)
        input_size=28,   # Input image size
        X=64,            # Number of filters in conv1
        Y=32,            # Number of filters in conv2
        Z=64,            # Number of neurons in fc1
        num_classes=10,  # Number of output classes
    ):
        
        super().__init__()

        self.Y = Y
        self.Z = Z

        self.conv1 = nn.Conv2d(
            in_channels,
            X,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        if Y > 0:
            self.conv2 = nn.Conv2d(
                X,
                Y,
                kernel_size=3,
                padding=1
            )
            channels = Y
            spatial = input_size // 4
        else:
            self.conv2 = None
            channels = X
            spatial = input_size // 2

        self.flatten_dim = channels * spatial * spatial

        if Z > 0:
            self.fc1 = nn.Linear(self.flatten_dim, Z)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(Z, num_classes)
        else:
            self.fc1 = None
            self.fc2 = nn.Linear(self.flatten_dim, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        if self.conv2 is not None:
            x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        if self.fc1 is not None:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)

        x = self.fc2(x)

        return x