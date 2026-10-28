"""ConvLSTM forecaster (Shi et al., 2015, no peepholes) and reference predictors."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """[i, f, o, g] = Conv([u_t, h_{t-1}]);  c = s(f)*c + s(i)*tanh(g);  h = s(o)*tanh(c)."""

    def __init__(self, in_ch, hid_ch, kernel=3, dilation=1):
        super().__init__()
        self.hid_ch = hid_ch
        pad = dilation * (kernel - 1) // 2          # 'same' size, zero padding
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, stride=1,
                              padding=pad, dilation=dilation, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """9-step rollout; steps 6-9 feed back own predictions; outputs of steps 5-9 returned."""

    def __init__(self, patch=4, hidden=(64, 64), kernels=(3, 3), dilations=(1, 1)):
        super().__init__()
        self.patch = patch
        in_ch = patch * patch
        cells, prev = [], in_ch
        for hc, k, d in zip(hidden, kernels, dilations):
            cells.append(ConvLSTMCell(prev, hc, k, d))
            prev = hc
        self.cells = nn.ModuleList(cells)
        self.head = nn.Conv2d(prev, in_ch, kernel_size=1, bias=True)

    def step(self, frame, states):
        u = F.pixel_unshuffle(frame, self.patch)
        new_states = []
        for cell, st in zip(self.cells, states):
            h, c = cell(u, st)
            new_states.append((h, c))
            u = h
        out = torch.sigmoid(F.pixel_shuffle(self.head(u), self.patch))
        return out, new_states

    def forward(self, X, n_out=5):
        """(B, 5, 1, 64, 64) -> (B, n_out, 1, 64, 64)."""
        B, T_in, C, H, W = X.shape
        gh, gw = H // self.patch, W // self.patch
        states = [(X.new_zeros(B, c.hid_ch, gh, gw), X.new_zeros(B, c.hid_ch, gh, gw))
                  for c in self.cells]
        preds, prev = [], None
        for tau in range(1, T_in + n_out):            # tau = 1..9
            u = X[:, tau - 1] if tau <= T_in else prev
            prev, states = self.step(u, states)
            if tau >= T_in:                            # outputs of steps 5..9
                preds.append(prev)
        return torch.stack(preds, dim=1)


def model_A():
    return ConvLSTMForecaster(patch=4, hidden=(64, 64), kernels=(3, 3), dilations=(1, 1))


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------- reference models
class Blank(nn.Module):
    def forward(self, X, n_out=5):
        B, _, C, H, W = X.shape
        return X.new_zeros(B, n_out, C, H, W)


class CopyLast(nn.Module):
    def forward(self, X, n_out=5):
        return X[:, -1:].expand(-1, n_out, -1, -1, -1).clone()
