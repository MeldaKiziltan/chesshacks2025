import torch
import torch.nn as nn
import torch.nn.functional as F
from .consts import ACTION_SIZE

class Anon(nn.Module):
    def __init__(self, in_channels=18, filters=64, action_size=ACTION_SIZE):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(filters)

        # 3 simple residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(3):
            self.res_blocks.append(nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters),
                nn.ReLU(),
                nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters),
            ))

        # policy head
        self.pol_conv = nn.Conv2d(filters, 32, kernel_size=1)
        self.pol_bn = nn.BatchNorm2d(32)
        self.pol_fc = nn.Linear(32 * 8 * 8, action_size)

        # value head
        self.val_conv = nn.Conv2d(filters, 16, kernel_size=1)
        self.val_bn = nn.BatchNorm2d(16)
        self.val_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for blk in self.res_blocks:
            out = blk(x)
            x = F.relu(x + out)

        # policy
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = p.view(p.size(0), -1)
        logits = self.pol_fc(p)

        # value
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.val_fc1(v))
        value = torch.tanh(self.val_fc2(v)).squeeze(-1)

        return logits, value
