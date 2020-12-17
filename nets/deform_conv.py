import torch
from torch import nn
from detectron2.layers import ModulatedDeformConv


class DCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1):
        super(DCN, self).__init__()

        channels_ = deformable_groups * 3 * kernel_size * kernel_size  # only support square kernels
        self.conv_offset_mask = nn.Conv2d(in_channels,
                                          channels_,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          bias=True)
        self.deform_conv = ModulatedDeformConv(in_channels, out_channels, kernel_size, stride,
                                               padding, dilation, deformable_groups=deformable_groups)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, inputs):
        out = self.conv_offset_mask(inputs)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)

        return self.deform_conv(inputs, offset, mask)






