from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, projections: nn.Module = None):
        """
            A Basic Block for ResNet.

            params:
                - in_channels (int): Number of input channels.
                - out_channels (int): Number of output channels.
                - stride (int, optional): Stride for 1st convolutional layer. Defaults to 1, set 2 to for downsampling.
                - projections (nn.Module, optional): Aligns dimensions of input and output for shortcut connection.
        """

        super(BasicBlock, self).__init__()
        self.expansion = 1  # number of channels preserved across all convolutional layers in BasicBlock
        self.projections = projections

        # for the 1st convolutional layer, stride may be 1 or 2 (see docstring)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        # Note: Bias redundant due to follow-up BN layer.
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, bias=False)  # stride 1 by default
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)  # Note: PyTorch and other implementations have 'inplace=True'

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


class Bottleneck(nn.Module):
    """
        A Bottleneck block for deep ResNet architectures (e.g., ResNet-50, ResNet-101).

        The Bottleneck block reduces computational cost by using 1x1 convolutions to reduce and increase the channel
        dimensions before and after a 3x3 convolution.

        params:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels before expansion.
            - stride (int, optional): Stride for the first convolution layer. Defaults to 1.
            - projections (nn.Module, optional): Projection layer for aligning dimensions in the residual connection.
    """

    def __init__(self, in_channels, out_channels, stride=1, projections=None):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.projections = projections

        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=stride, bias=False)
        # Note: According to the original authors downsampling is always performed at the first convolutional layer
        # To mimic PyTorch's version shift the stride argument to the 3x3 conv layer
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv with expansion
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.projections is not None:
            identity = self.projections(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
        A ResNet model implementation.

        Args:
            block (nn.Module): Block type (BasicBlock or Bottleneck).
            layers (list[int]): Number of blocks in each layer.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            color_channels (int, optional): Number of input color channels. Defaults to 3.
    """

    def __init__(self, block, layers, num_classes=1000, color_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # adjust color_channels for this layer to 1 for grayscale images
        self.conv1 = nn.Conv2d(in_channels=color_channels, out_channels=self.in_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(block, 64, layers[0])
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layers(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # nn.AvgPool2d() can also be used
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

        self._initalize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # fan_out helps stabilize gradient flow during backpropagation for very deep networks
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # Default initialization, ensures that BN behaves like identity initially
                nn.init.constant_(m.weight, val=1)  # gamma
                nn.init.constant_(m.bias, val=0)  # beta

    def _make_layers(self, block, out_channels, num_blocks, stride=1):
        projections = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            projections = nn.Sequential(
                # 1x1 conv
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, projections)]

        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_channels, out_channels)
            )

        return nn.Sequential(*layers)
