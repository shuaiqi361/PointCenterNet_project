import torch
import torch.nn as nn


class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj * 2 + 1)

    def forward(self, inputs):
        inputs = torch.cat([inputs[..., -self.n_adj:], inputs, inputs[..., :self.n_adj]], dim=2)
        return self.fc(inputs)


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj * 2 + 1, dilation=self.dilation)

    def forward(self, inputs):
        if self.n_adj != 0:
            inputs = torch.cat(
                [inputs[..., -self.n_adj * self.dilation:], inputs, inputs[..., :self.n_adj * self.dilation]], dim=2)
        return self.fc(inputs)


_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}


class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type='dgrid', n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', n_adj=4):
        super(Snake, self).__init__()
        self.n_adj = n_adj
        self.head = BasicBlock(feature_dim, state_dim, conv_type, n_adj=self.n_adj)

        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        # for i in range(self.res_layer_num):
        #     self.conv.append(BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[i]))
        self.conv1 = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[0])
        self.conv2 = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[1])
        self.conv3 = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[2])
        self.conv4 = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[3])
        self.conv5 = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[4])
        self.conv6 = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[5])
        self.conv7 = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[6])

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x):
        # states = []
        x = self.head(x)
        # states.append(x)
        # for i in range(self.res_layer_num):
        #     x = self.__getattr__('res' + str(i))(x, adj) + x
        #     states.append(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

        state = torch.cat([x, x1, x2, x3, x4, x5, x6, x7], dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x
