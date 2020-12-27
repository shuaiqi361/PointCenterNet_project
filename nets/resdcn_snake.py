import math
import torch
import torch.nn as nn
from nets.deform_conv import DCN
from nets.snake_conv import Snake
import torch.utils.model_zoo as model_zoo

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


class SnakeResDCN(nn.Module):
    def __init__(self, block, layers, head_conv=64, num_classes=80, snake_adj=4, dictionary=None):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.num_classes = num_classes
        self.snake_adj = snake_adj
        self.max_obj = 128
        self.n_vertices = 32
        self.dict_tensor = dictionary

        self.dict_tensor.requires_grad = False

        super(SnakeResDCN, self).__init__()
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

        # used for snake conv on offsets
        self.snake_1 = Snake(state_dim=64, feature_dim=64 + 2, n_adj=self.snake_adj)
        self.snake_2 = Snake(state_dim=64, feature_dim=64 + 2, n_adj=self.snake_adj)
        self.snake_3 = Snake(state_dim=64, feature_dim=64 + 2, n_adj=self.snake_adj)

        if head_conv > 0:
            # heatmap layers
            self.hmap = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True))
            self.hmap[-1].bias.data.fill_(-2.19)
            # regression layers
            self.regs = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
            self.w_h_ = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, 4, kernel_size=1, bias=True))
            self.codes = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(head_conv),
                                      nn.Conv2d(head_conv, 64, kernel_size=1, bias=True))
        else:
            raise NotImplementedError

        fill_fc_weights(self.regs)
        fill_fc_weights(self.w_h_)
        fill_fc_weights(self.codes)

    def get_vertex_features(self, fmaps, poly):
        """
        :param fmaps: (N, C, H_in, W_in)
        :param poly: (N, max_obj, n_vertices, 2)
        :param h: spatial dim: width and height of the last feature map, 512 x 512
        :param w:
        :return: interpolated features (N, C, max_obj, n_vertices) --> (N, max_obj, C, n_vertices)
        """
        bs, c, h, w = fmaps.size()
        obj_polygons = poly.clone()
        obj_polygons[..., 0] = obj_polygons[..., 0] / (w / 2.) - 1  # the grid argument is in the range [-1, 1]
        obj_polygons[..., 1] = obj_polygons[..., 1] / (h / 2.) - 1

        vert_feature = nn.functional.grid_sample(fmaps, grid=obj_polygons, mode='bilinear')  # (bs, C, max_obj, n_vertices)

        return vert_feature.permute(0, 2, 1, 3).contiguous()

    def _gather_feature(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feature(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feature(feat, ind)
        return feat

    def _nms(self, heat, kernel=3):
        hmax = nn.functional.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, k=128):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
        topk_clses = (topk_ind / k).int()
        topk_inds = self._gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_ys = self._gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_xs = self._gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

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

    def forward(self, x, inds=None, gt_center=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        fmap = self.deconv_layers(x)
        hmap_out = self.hmap(fmap)
        regs_out = self.regs(fmap)
        w_h_out = self.w_h_(fmap)
        codes_out = self.codes(fmap)

        bs, ch, h, w = fmap.size()
        if inds is None:
            hmap_out = self._nms(hmap_out)  # perform nms on heatmaps
            _, inds, _, ys, xs = self._topk(hmap_out, k=self.max_obj)

        # extract vertices features from the fmap
        obj_codes = self._tranpose_and_gather_feature(codes_out, inds)
        obj_regs = self._tranpose_and_gather_feature(regs_out, inds)

        if gt_center is None:
            obj_regs = obj_regs.view(bs, self.max_obj, 2)
            xs = xs.view(bs, self.max_obj, 1) + obj_regs[:, :, 0:1]
            ys = ys.view(bs, self.max_obj, 1) + obj_regs[:, :, 1:2]
            gt_center = torch.cat([xs, ys], dim=2)

        # obj_codes = self._tranpose_and_gather_feature(obj_codes, inds)
        obj_codes = obj_codes.view(bs, self.max_obj, 64)

        segms = torch.matmul(obj_codes, self.dict_tensor)
        polys = segms.view(bs, self.max_obj, 32, 2) + gt_center.view(bs, self.max_obj, 1, 2)

        # first snake
        vertex_feats = self.get_vertex_features(fmap, polys)  # (N, max_obj, C, 32)
        vertex_feats = torch.cat([vertex_feats, polys - gt_center.view(bs, self.max_obj, 1, 2)], dim=-2).detach()
        batch_v_feats = []
        for n in range(bs):
            batch_v_feats.append(self.snake_1(vertex_feats[n]).unsqueeze(0))

        offsets = torch.cat(batch_v_feats, dim=0)  # (N, 2, max_obj, 32)
        polys_1 = polys + offsets.permute(0, 1, 3, 2).contiguous()

        # second snake
        vertex_feats = self.get_vertex_features(fmap, polys_1)  # (N, max_obj, C, 32)
        vertex_feats = torch.cat([vertex_feats, polys_1 - gt_center.view(bs, self.max_obj, 1, 2)], dim=-2).detach()
        batch_v_feats = []
        for n in range(bs):
            batch_v_feats.append(self.snake_2(vertex_feats[n]).unsqueeze(0))

        offsets = torch.cat(batch_v_feats, dim=0)  # (N, 2, max_obj, 32)
        polys_2 = polys_1 + offsets.permute(0, 1, 3, 2).contiguous()

        # third snake
        vertex_feats = self.get_vertex_features(fmap, polys_2)  # (N, max_obj, C, 32)
        vertex_feats = torch.cat([vertex_feats, polys_2 - gt_center.view(bs, self.max_obj, 1, 2)], dim=-2).detach()
        batch_v_feats = []
        for n in range(bs):
            batch_v_feats.append(self.snake_3(vertex_feats[n]).unsqueeze(0))

        offsets = torch.cat(batch_v_feats, dim=0)  # (N, 2, max_obj, 32)
        polys_3 = polys_2 + offsets.permute(0, 1, 3, 2).contiguous()

        out = [[hmap_out, regs_out, w_h_out, codes_out, polys_1, polys_2, polys_3]]
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


def get_pose_resdcn(num_layers=50, head_conv=64, num_classes=80, dictionary=None):
    block_class, layers = resnet_spec[num_layers]
    model = SnakeResDCN(block_class, layers, head_conv, num_classes, dictionary=dictionary)
    model.init_weights(num_layers)
    return model


if __name__ == '__main__':
    import torch
    from collections import OrderedDict
    from utils.utils import count_parameters, count_flops


    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
        # pass


    net = get_pose_net(18, num_classes=80).cuda()

    # ckpt = torch.load('../ckpt/ctdet_pascal_resdcn18_384.pth')['state_dict']
    # new_ckpt = OrderedDict()
    # for k in ckpt:
    #   new_ckpt[k.replace('hm', 'hmap').replace('wh', 'w_h_').replace('reg', 'regs')] = ckpt[k]
    # torch.save(new_ckpt, '../ckpt/resdcn18_baseline/checkpoint.t7')
    # net.load_state_dict(new_ckpt)

    # count_parameters(net)
    # count_flops(net, input_size=512)

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, DCN):
            m.register_forward_hook(hook)

    with torch.no_grad():
        y = net(torch.randn(2, 3, 512, 512).cuda())
