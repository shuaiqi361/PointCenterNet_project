import torch
import torch.nn as nn


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class VotingModule(nn.Module):
    def __init__(self, inplanes, head_conv, outplanes):
        super(VotingModule, self).__init__()
        self.input_conv = nn.Sequential(nn.Conv2d(inplanes, head_conv, kernel_size=1, padding=0, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(head_conv, head_conv, kernel_size=1, padding=0, bias=True))

        self.votes_conv = nn.Sequential(nn.Conv2d(head_conv, head_conv, kernel_size=1, padding=0, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(head_conv, outplanes, kernel_size=1, padding=0, bias=True))

        fill_fc_weights(self.input_conv)
        fill_fc_weights(self.votes_conv)

    def forward(self, x, votes):
        x = self.input_conv(votes) * x
        x = self.votes_conv(x)

        return x
