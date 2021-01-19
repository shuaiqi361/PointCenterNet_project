import torch
import torch.nn as nn


class SpatialAggregationModule(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=[4, 8, 12], padding=[4, 8, 12]):
        super(SpatialAggregationModule, self).__init__()
        self.stride = stride
        self.padding = padding  # padding should be the same as dilation, so the output plane is the same as input
        self.dilation = dilation
        self.n_path = len(dilation)
        self.d1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(planes, planes, kernel_size=3, dilation=self.dilation[0], padding=self.padding[0], stride=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(planes, inplanes, kernel_size=1, padding=0, bias=False),
                                nn.BatchNorm2d(inplanes),
                                nn.ReLU(inplace=True))
        self.d2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(planes, planes, kernel_size=3, dilation=self.dilation[1],
                                          padding=self.padding[1], stride=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(planes, inplanes, kernel_size=1, padding=0, bias=False),
                                nn.BatchNorm2d(inplanes),
                                nn.ReLU(inplace=True))
        self.d3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(planes, planes, kernel_size=3, dilation=self.dilation[2],
                                          padding=self.padding[2], stride=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(planes, inplanes, kernel_size=1, padding=0, bias=False),
                                nn.BatchNorm2d(inplanes),
                                nn.ReLU(inplace=True))

        self.conv_aft = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=1, padding=0, bias=False),
                                      nn.BatchNorm2d(inplanes),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(inplanes),
                                      nn.ReLU(inplace=True))

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(x)
        d3 = self.d3(x)

        out = x + d1 + d2 + d3
        out = self.conv_aft(out)

        return out
