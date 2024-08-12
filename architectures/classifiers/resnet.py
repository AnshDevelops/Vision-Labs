from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, stride=1, projections=None):
        """
            A Basic Block for ResNet.

            params:
                - in_channels (int): Number of input channels.
                - intermediate_channels (int): Number of channels in the intermediate output.
                - out_channels (int): Number of output channels.
                - stride (int, optional): Stride for 1st convolutional layer. Defaults to 1, set 2 to for downsampling.
                - projections (nn.Module, optional): Aligns dimensions of input and output for shortcut connection.
            """

        super(BasicBlock, self).__init__()
        self.expansion = 1  # number of channels preserved across all convolutional layers in a block

        # for the 1st convolutional layer, stride may be 1 or 2 (see docstring)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)  # Bias redundant due to follow-up BN layer
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)  # Note: PyTorch and other implementations have 'inplace=True'

        self.conv2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, bias=False)  # stride 1 by default
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projections = projections

    def forward(self, x):
        identity = x  # to be added using a shortcut connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.projections is not None:
            identity = self.projections(x)  # Adding projections to identity to match dimensions with output

        out += identity
        out = self.relu(out)
        return out


# class BottleneckBlock(nn.Module):
#     def __init__(self, in_channels, intermediate_channels, out_channels, stride=1, projections=None):
#         super(BottleneckBlock, self).__init__()
#
#         self.expansion = 4
#         self.projections = projections


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=7, stride=2,
                               padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # nn.AvgPool2d() can also be used
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    # def _make_layers(self):
