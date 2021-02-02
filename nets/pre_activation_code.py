import torch
import torch.nn as nn


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PreActResidualModule(nn.Module):
    def __init__(self, inplanes, head_conv, outplanes, num_blocks):
        super(PreActResidualModule, self).__init__()
        self.residual_convs = []
        self.output_convs = []
        self.input_conv = nn.Sequential(nn.Conv2d(inplanes, head_conv, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm2d(head_conv),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=False))

        for i in range(num_blocks):
            self.residual_convs.append(nn.Sequential(nn.BatchNorm2d(head_conv),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(head_conv, head_conv // 2, kernel_size=1, padding=0, bias=False),
                                                     nn.BatchNorm2d(head_conv // 2),
                                                     nn.ReLU(inplace=True),
                                                     nn.Conv2d(head_conv // 2, head_conv, kernel_size=3, padding=1, bias=False)))
            self.output_convs.append(nn.Sequential(nn.BatchNorm2d(head_conv),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv2d(head_conv, outplanes, kernel_size=1, padding=0, bias=True)))

        for l in self.residual_convs:
            fill_fc_weights(l)

        for l in self.output_convs:
            fill_fc_weights(l)

    def forward(self, x):
        x = self.input_conv(x)

        out = []
        for i in range(len(self.output_convs)):
            x = x + self.residual_convs[i](x)
            out.append(self.output_convs[i](x))

        return out
