import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from nets.spatial_aggregation_conv_large import SpatialAggregationModule


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                                  nn.BatchNorm2d(out_dim)) \
            if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


# inp_dim -> out_dim -> ... -> out_dim
def make_layer(kernel_size, inp_dim, out_dim, modules, layer, stride=1):
    layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]
    layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
    return nn.Sequential(*layers)


# inp_dim -> inp_dim -> ... -> inp_dim -> out_dim
def make_layer_revr(kernel_size, inp_dim, out_dim, modules, layer):
    layers = [layer(kernel_size, inp_dim, inp_dim) for _ in range(modules - 1)]
    layers.append(layer(kernel_size, inp_dim, out_dim))
    return nn.Sequential(*layers)


# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                         nn.Conv2d(curr_dim, out_dim, (1, 1)))


class kp_module(nn.Module):
    def __init__(self, n, dims, modules):
        super(kp_module, self).__init__()

        self.n = n

        curr_modules = modules[0]
        next_modules = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        # curr_mod x residual，curr_dim -> curr_dim -> ... -> curr_dim
        self.top = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual)
        self.down = nn.Sequential()
        # curr_mod x residual，curr_dim -> next_dim -> ... -> next_dim
        self.low1 = make_layer(3, curr_dim, next_dim, curr_modules, layer=residual, stride=2)
        # next_mod x residual，next_dim -> next_dim -> ... -> next_dim
        if self.n > 1:
            self.low2 = kp_module(n - 1, dims[1:], modules[1:])
        else:
            self.low2 = make_layer(3, next_dim, next_dim, next_modules, layer=residual)
        # curr_mod x residual，next_dim -> next_dim -> ... -> next_dim -> curr_dim
        self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, layer=residual)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = self.top(x)
        down = self.down(x)
        low1 = self.low1(down)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up(low3)
        return up1 + up2


class exkp(nn.Module):
    def __init__(self, n, nstack, dims, modules, cnv_dim=256, num_classes=80, n_codes=64):
        super(exkp, self).__init__()

        self.nstack = nstack
        self.num_classes = num_classes
        self.n_codes = n_codes

        curr_dim = dims[0]

        self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                                 residual(3, 128, curr_dim, stride=2))

        self.kps = nn.ModuleList([kp_module(n, dims, modules) for _ in range(nstack)])

        self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])

        self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])

        self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                                                    nn.BatchNorm2d(curr_dim))
                                      for _ in range(nstack - 1)])
        self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                                                  nn.BatchNorm2d(curr_dim))
                                    for _ in range(nstack - 1)])
        # heatmap layers
        self.hmap = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
        for hmap in self.hmap:
            hmap[-1].bias.data.fill_(-2.19)

        # regression layers from inmodal features
        self.regs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
        self.w_h_ = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])

        # codes layers from amodal features
        self.codes = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, self.n_codes) for _ in range(nstack)])
        self.offsets = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        inter = self.pre(image)

        outs = []
        for ind in range(self.nstack):
            kp = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp)

            if self.training or ind == self.nstack - 1:
                outs.append([self.hmap[ind](cnv), self.regs[ind](cnv), self.w_h_[ind](cnv),
                             self.codes[ind](cnv), self.offsets[ind](cnv)])

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs


get_hourglass = \
    {'large_hourglass':
         exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4], n_codes=64),
     'small_hourglass':
         exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4], n_codes=64)}

if __name__ == '__main__':
    from collections import OrderedDict
    from utils.utils import count_parameters, count_flops, load_model


    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
        # pass


    net = get_hourglass['large_hourglass']
    load_model(net, '../ckpt/pretrain/checkpoint.t7')
    count_parameters(net)
    count_flops(net, input_size=512)

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(hook)

    with torch.no_grad():
        y = net(torch.randn(2, 3, 512, 512).cuda())
    # print(y.size())
