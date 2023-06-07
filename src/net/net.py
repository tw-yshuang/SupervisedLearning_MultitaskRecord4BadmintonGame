from typing import List
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from src.net.EfficientNetV2_M import efficientnet_v2_m as EffNet

class LinNet(nn.Module):
    def __init__(self, input, output_classes, isOneHot=False):
        super(LinNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_classes),
        )

        self.last = nn.Softmax(dim=1) if isOneHot else self.none_func

    @staticmethod
    def none_func(x: torch.Tensor):
        return x

    def forward(self, x):
        return self.last(self.linear(x))


class BadminationNet(nn.Module):
    def __init__(self, in_seq):
        super(BadminationNet, self).__init__()

        eff_out = 2048
        self.eff = EffNet(in_seq=in_seq, output_classes=eff_out)
        self.lins = nn.ModuleList(
            [
                LinNet(eff_out, 6, isOneHot=True),  # 0~6
                LinNet(eff_out, 2, isOneHot=True),  # 6~8
                LinNet(eff_out, 2, isOneHot=True),  # 8~10
                LinNet(eff_out, 2, isOneHot=True),  # 10~12
                LinNet(eff_out, 2, isOneHot=True),  # 12~14
                LinNet(eff_out, 6, isOneHot=False),  # 14~20
                LinNet(eff_out, 9, isOneHot=True),  # 20~29
                LinNet(eff_out, 3, isOneHot=True),  # 29~32
            ]
        )

        self.init_loss_funcs()
        self.init_optims()

    def init_loss_funcs(self):
        self.cn = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def init_optims(self):
        self.eff_optim = optim.AdamW(self.eff.parameters(), lr=0.0001)
        self.lin_optims = [optim.AdamW(lin.parameters(), lr=0.0001) for lin in self.lins]
        # self.eff_optim = optim.Adam(self.eff.parameters(), lr=0.001)
        # self.lin_optims = [optim.SGD(lin.parameters(), lr=0.001) for lin in self.lins]

    def forward(self, x):
        x = self.eff(x)
        return torch.hstack([lin(x) for lin in (self.lins)])

    def update(self, pred: torch.Tensor, labels: torch.Tensor, isTrain=True):
        loss_record = 0.0

        idxs = [6, 8, 10, 12, 14, 20, 29, 32]
        loss_func_order: List[nn.Module] = [*[self.cn] * 5, self.mse, *[self.cn] * 2]

        idx_start = 0
        for idx_end, loss_func, lin_optim in zip(idxs, loss_func_order, self.lin_optims):
            if isTrain:
                loss: torch.Tensor = loss_func(pred[:, idx_start:idx_end], labels[:, idx_start:idx_end])
                loss.backward(retain_graph=True)
                lin_optim.step()
                lin_optim.zero_grad()
            else:
                with torch.no_grad():
                    loss: torch.Tensor = loss_func(pred[:, idx_start:idx_end], labels[:, idx_start:idx_end])

            loss_record += loss.item()
            idx_start = idx_end

        if isTrain:
            self.eff_optim.step()
            self.eff_optim.zero_grad()

        return loss_record


if __name__ == '__main__':
    bad_net = BadminationNet(5).to('cuda:0')

    for _ in range(10):
        aa = bad_net(torch.randn((2, 5, 3, 512, 512)).to('cuda:0'))

        cc = torch.tensor([[*[0.0] * 5, 1.0, *[0.0, 1.0] * 4, *torch.randn(6), *[0.0] * 8, 1.0, *[0.0] * 2, 1.0]] * 2).to('cuda:0')

        bad_net.update(aa, cc)
