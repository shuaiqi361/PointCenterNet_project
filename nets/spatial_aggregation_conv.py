import torch
import torch.nn as nn


class SpatialAggregationModule(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=[1, 6, 12, 18], padding=[1, 6, 12, 18]):
        super(SpatialAggregationModule, self).__init__()
        self.stride = stride
        self.padding = padding  # padding should be the same as dilation, so the output plane is the same as input
        self.dilation = dilation
        self.n_path = len(dilation)
        self.shared_conv1 = nn.Parameter(torch.randn(planes, inplanes, 1, 1))  # shrink the channels
        self.shared_conv2 = nn.Parameter(torch.randn(planes, planes, 3, 3))
        self.shared_conv3 = nn.Parameter(torch.randn(inplanes, planes, 1, 1))

        self.bn11 = nn.BatchNorm2d(planes)
        self.bn12 = nn.BatchNorm2d(planes)
        self.bn13 = nn.BatchNorm2d(inplanes)

        self.bn21 = nn.BatchNorm2d(planes)
        self.bn22 = nn.BatchNorm2d(planes)
        self.bn23 = nn.BatchNorm2d(inplanes)

        self.bn31 = nn.BatchNorm2d(planes)
        self.bn32 = nn.BatchNorm2d(planes)
        self.bn33 = nn.BatchNorm2d(inplanes)

        self.bn41 = nn.BatchNorm2d(planes)
        self.bn42 = nn.BatchNorm2d(planes)
        self.bn43 = nn.BatchNorm2d(inplanes)

        self.relu = nn.ReLU(inplace=True)

        self.conv_aft = nn.Sequential(nn.Conv2d(inplanes * 4, inplanes, kernel_size=1, padding=0, bias=False),
                                      nn.BatchNorm2d(inplanes),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(inplanes),
                                      nn.ReLU(inplace=True))

    def forward_d1(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.shared_conv1, bias=None)
        out = self.relu(self.bn11(out))

        out = nn.functional.conv2d(out, self.shared_conv2, stride=self.stride, padding=self.padding[0],
                                   dilation=self.dilation[0], bias=None)
        out = self.relu(self.bn12(out))

        out = nn.functional.conv2d(out, self.shared_conv3, bias=None)
        out = self.relu(self.bn13(out))

        out = self.relu(out + residual)

        return out

    def forward_d2(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.shared_conv1, bias=None)
        out = self.relu(self.bn21(out))

        out = nn.functional.conv2d(out, self.shared_conv2, stride=self.stride, padding=self.padding[1],
                                   dilation=self.dilation[1], bias=None)
        out = self.relu(self.bn22(out))

        out = nn.functional.conv2d(out, self.shared_conv3, bias=None)
        out = self.relu(self.bn23(out))

        out = self.relu(out + residual)

        return out

    def forward_d3(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.shared_conv1, bias=None)
        out = self.relu(self.bn31(out))

        out = nn.functional.conv2d(out, self.shared_conv2, stride=self.stride, padding=self.padding[2],
                                   dilation=self.dilation[2], bias=None)
        out = self.relu(self.bn32(out))

        out = nn.functional.conv2d(out, self.shared_conv3, bias=None)
        out = self.relu(self.bn33(out))

        out = self.relu(out + residual)

        return out

    def forward_d4(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.shared_conv1, bias=None)
        out = self.relu(self.bn41(out))

        out = nn.functional.conv2d(out, self.shared_conv2, stride=self.stride, padding=self.padding[3],
                                   dilation=self.dilation[3], bias=None)
        out = self.relu(self.bn42(out))

        out = nn.functional.conv2d(out, self.shared_conv3, bias=None)
        out = self.relu(self.bn43(out))

        out = self.relu(out + residual)

        return out

    def forward(self, x):
        d1 = self.forward_d1(x)
        d2 = self.forward_d2(x)
        d3 = self.forward_d3(x)
        d4 = self.forward_d4(x)

        out = torch.cat([d1, d2, d3, d4], dim=1)
        out = self.conv_aft(out)

        return out

