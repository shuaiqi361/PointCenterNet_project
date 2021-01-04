import os
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', }


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PyramidFeatures(nn.Module):
    def __init__(self, c3_size, c4_size, c5_size, c6_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P6_1 = nn.Conv2d(c6_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P6_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P6_upsampled = nn.ConvTranspose2d(in_channels=feature_size,
                                out_channels=feature_size,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=False)
        fill_up_weights(self.P6_upsampled)

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(c5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_upsampled = nn.ConvTranspose2d(in_channels=feature_size,
                                out_channels=feature_size,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=False)
        fill_up_weights(self.P5_upsampled)
        # self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(c4_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_upsampled = nn.ConvTranspose2d(in_channels=feature_size,
                                out_channels=feature_size,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=False)
        fill_up_weights(self.P4_upsampled)
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(c3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        # self.P6 = nn.Conv2d(c5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, c3, c4, c5, c6):
        P6_x = self.P6_1(c6)
        P6_upsampled_x = self.P6_upsampled(P6_x)

        P5_x = self.P5_1(c5)
        P5_x = P6_upsampled_x + P5_x
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(c4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(c3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.relu(self.P3_2(P3_x))

        # P6_x = self.P6(c5)
        #
        # P7_x = self.P7_1(P6_x)
        # P7_x = self.P7_2(P7_x)

        # return [P3_x, P4_x, P5_x, P6_x, P7_x]
        return P3_x

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.normal_(m.weight, std=0.001)
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class FPNResNet(nn.Module):
    def __init__(self, block, layers, head_conv, num_classes):
        super(FPNResNet, self).__init__()
        self.inplanes = 64
        self.deconv_with_bias = False
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        # self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])
        self.fpn_layers = PyramidFeatures(256, 512, 1024, 2048, feature_size=256)

        if head_conv > 0:
            # heatmap layers
            self.hmap = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True))
            self.hmap[-1].bias.data.fill_(-2.19)
            # regression layers
            self.regs = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
            self.w_h_ = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, 4, kernel_size=1, bias=True))

            self.codes_1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(128),
                                         nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=True))
            self.compress_1 = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(64),
                                         nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=True))
            self.codes_2 = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(64),
                                         nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(128),
                                         nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=True))
            self.compress_2 = nn.Sequential(nn.ReLU(inplace=True),
                                            nn.BatchNorm2d(64),
                                            nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=True))
            self.codes_3 = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(64),
                                         nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(128),
                                         nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=True))
            self.compress_3 = nn.Sequential(nn.ReLU(inplace=True),
                                            nn.BatchNorm2d(64),
                                            nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=True))

        fill_fc_weights(self.regs)
        fill_fc_weights(self.w_h_)
        fill_fc_weights(self.codes_1)
        fill_fc_weights(self.codes_2)
        fill_fc_weights(self.codes_3)
        fill_fc_weights(self.compress_1)
        fill_fc_weights(self.compress_2)
        fill_fc_weights(self.compress_3)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # print(x1.size(), x2.size(), x3.size(), x4.size())

        x = self.fpn_layers(x1, x2, x3, x4)

        xc_1 = self.compress_1(self.codes_1(x))
        xc_2 = self.compress_2(self.codes_2(xc_1) + xc_1)
        xc_3 = self.compress_3(self.codes_3(xc_2) + xc_2)

        out = [[self.hmap(x), self.regs(x), self.w_h_(x), xc_1, xc_2, xc_3]]

        # out = [[self.hmap(x), self.regs(x), self.w_h_(x), self.codes_(x)]]
        return out

    def init_weights(self, num_layers):
        url = model_urls['resnet{}'.format(num_layers)]
        pretrained_state_dict = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_fpn_resnet(num_layers, head_conv, num_classes=80):
    block_class, layers = resnet_spec[num_layers]

    model = FPNResNet(block_class, layers, head_conv=head_conv, num_classes=num_classes)
    model.init_weights(num_layers)
    return model
