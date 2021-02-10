import math
# import torch
import torch.nn as nn
from nets.deform_conv import DCN
import torch.utils.model_zoo as model_zoo
from nets.voting_conv import VotingModule

BN_MOMENTUM = 0.1
model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


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
            # nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PoseResNet(nn.Module):
    def __init__(self, block, layers, head_conv, num_classes, num_codes, num_votes=121):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.num_classes = num_classes
        self.num_codes = num_codes

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(3, [256, 128, 64], [4, 4, 4])
        self.amodal_conv = nn.Sequential(nn.Conv2d(head_conv, head_conv * 2, kernel_size=3, padding=1, bias=True),
                                         nn.BatchNorm2d(head_conv * 2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(head_conv * 2, head_conv, kernel_size=3, padding=1, bias=True),
                                         nn.BatchNorm2d(head_conv),
                                         nn.ReLU(inplace=True))
        self.inmodal_conv = nn.Sequential(nn.Conv2d(head_conv, head_conv * 2, kernel_size=3, padding=1, bias=True),
                                         nn.BatchNorm2d(head_conv * 2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(head_conv * 2, head_conv, kernel_size=3, padding=1, bias=True),
                                         nn.BatchNorm2d(head_conv),
                                         nn.ReLU(inplace=True))

        if head_conv > 0:
            # heatmap layers
            self.hmap = nn.Sequential(nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True))
            self.hmap[-1].bias.data.fill_(-2.19)
            # regression layers
            self.regs = nn.Sequential(nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
            self.w_h_ = nn.Sequential(nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
            self.offsets = nn.Sequential(nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
            self.votes = nn.Sequential(nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(head_conv, num_votes, kernel_size=1, bias=True))

            # -------- inmodal features
            self.occ = nn.Sequential(nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(head_conv),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=3, dilation=3, bias=True),
                                     nn.BatchNorm2d(head_conv),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(head_conv),
                                     nn.ReLU(inplace=True))

            self.codes = nn.Sequential(nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(head_conv, num_codes, kernel_size=1, bias=True))

            self.voted_hmap = VotingModule(num_votes, num_classes, num_classes)
            self.voted_codes = VotingModule(num_votes, num_codes, num_codes)

        fill_fc_weights(self.regs)
        fill_fc_weights(self.w_h_)
        fill_fc_weights(self.occ)
        fill_fc_weights(self.offsets)
        fill_fc_weights(self.inmodal_conv)
        fill_fc_weights(self.amodal_conv)
        fill_fc_weights(self.votes)
        fill_fc_weights(self.codes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes,
                     kernel_size=3, stride=1,
                     padding=1, dilation=1, deformable_groups=1)

            up = nn.ConvTranspose2d(in_channels=planes,
                                    out_channels=planes,
                                    kernel_size=kernel,
                                    stride=2,
                                    padding=padding,
                                    output_padding=output_padding,
                                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        amodal_x = self.amodal_conv(x)
        inmodal_x = self.inmodal_conv(x)

        offsets = self.offsets(inmodal_x)
        votes = self.votes(inmodal_x)

        in_cls = self.occ(amodal_x + inmodal_x)
        hmap = self.hmap(in_cls)
        codes = self.codes(in_cls)

        voted_hmap = self.voted_hmap(hmap, votes.detach())
        voted_codes = self.voted_codes(codes, votes.detach())

        out = [[voted_hmap, self.regs(inmodal_x), self.w_h_(inmodal_x), voted_codes, offsets, votes]]
        return out

    def init_weights(self, num_layers):
        url = model_urls['resnet{}'.format(num_layers)]
        pretrained_state_dict = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)
        print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_resdcn(num_layers, head_conv=64, num_classes=80, num_codes=64, num_votes=121):
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers, head_conv, num_classes, num_codes=num_codes, num_votes=num_votes)
    model.init_weights(num_layers)

    return model
