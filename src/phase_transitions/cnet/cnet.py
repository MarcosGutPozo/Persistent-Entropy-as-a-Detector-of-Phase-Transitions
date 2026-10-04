import torch
import torch.nn as nn
import torch.nn.functional as F


class CNet(nn.Module):

    def __init__(
        self,
        in_channels=3,   # Input channels (3 for RGB)
        input_size=32,   # Input image size
        X=64,            # Number of filters in conv1
        Y=32,            # Number of filters in conv2
        Z=64,            # Number of neurons in fc1
        num_classes=10,  # Number of output classes
    ):

        super().__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels,
            X,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )

        self.lrn1 = nn.LocalResponseNorm(4)

        # Optional second convolution
        if Y > 0:
            self.conv2 = nn.Conv2d(
                X,
                Y,
                kernel_size=3,
                padding=1
            )
            channels = Y
        else:
            self.conv2 = None
            channels = X

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=1
        )

        self.lrn2 = nn.LocalResponseNorm(4)

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                in_channels,
                input_size,
                input_size
            )

            dummy = self.pool1(F.relu(self.conv1(dummy)))
            dummy = self.lrn1(dummy)

            if self.conv2 is not None:
                dummy = self.pool2(F.relu(self.conv2(dummy)))
                dummy = self.lrn2(dummy)

            flatten_dim = dummy.numel()

        # Optional fully connected hidden layer
        if Z > 0:

            self.fc1 = nn.Linear(flatten_dim, Z)
            self.fc2 = nn.Linear(Z, num_classes)

        else:

            self.fc1 = None
            self.fc2 = nn.Linear(flatten_dim, num_classes)

    def forward(self, x):

        # First convolutional block
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.lrn1(x)

        # Second convolutional block
        if self.conv2 is not None:
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.lrn2(x)

        # Flatten feature maps
        x = torch.flatten(x, 1)

        # Optional hidden layer
        if self.fc1 is not None:
            x = F.relu(self.fc1(x))

        # Output logits
        x = self.fc2(x)

        return x