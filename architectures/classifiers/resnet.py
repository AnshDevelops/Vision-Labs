from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, stride=1, projections=None):
        super(BasicBlock, self).__init__()

        # for the 1st convolutional layer, stride is 1 by default, but should be set to 2 if downsampling is to be done
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)  # Bias redundant due to follow-up BN layer
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)  # PyTorch and other implementations have 'inplace=True'

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

        if self.projections:
            identity = self.projections(x)  # Adding projections to identity

        out += identity
        out = self.relu(out)
        return out
