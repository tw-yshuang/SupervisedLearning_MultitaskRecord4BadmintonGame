from typing import List
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from src.net.EfficientNetV2_M import EffNet


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


class BadmintonNet(nn.Module):
    def __init__(self, in_seq, num_frame: int, loss_func_order: List[nn.Module]):
        super(BadmintonNet, self).__init__()

        eff_out = 2048
        self.eff = EffNet(in_seq=in_seq, output_classes=eff_out)
        self.lins = nn.ModuleList(
            [
                LinNet(eff_out, num_frame + 1, isOneHot=True),  # 0~num_frame
                LinNet(eff_out, 2, isOneHot=True),  # -23~-21
                LinNet(eff_out, 2, isOneHot=True),  # -21~-19
                LinNet(eff_out, 2, isOneHot=True),  # -19~-17
                LinNet(eff_out, 2, isOneHot=True),  # -17~-15
                LinNet(eff_out, 9, isOneHot=True),  # -15~-6
                LinNet(eff_out, 6, isOneHot=False),  # -6~None
            ]
        )
        self.end_idx_orders = [-23, -21, -19, -17, -15, -6, None]
        self.loss_func_order = loss_func_order

        self.eff_optim: optim.Optimizer
        self.lin_optims: List[optim.Optimizer]
        self.eff_lr: float
        self.lin_lrs: List[float]

    # def init_loss_funcs(self):
    #     self.cn = nn.CrossEntropyLoss()
    #     self.mse = nn.MSELoss()

    def init_optims(self, eff_optim: optim.Optimizer, lin_optims: List[optim.Optimizer], eff_lr: float, lin_lrs: List[float]):
        self.eff_optim = eff_optim(self.eff.parameters(), lr=eff_lr)
        self.lin_optims = [optim(lin.parameters(), lr=lr) for optim, lr, lin in zip(lin_optims, lin_lrs, self.lins)]

        # self.eff_optim = optim.Adam(self.eff.parameters(), lr=0.001)
        # self.lin_optims = [optim.SGD(lin.parameters(), lr=0.001) for lin in self.lins]

    def forward(self, x):
        x = self.eff(x)
        return torch.hstack([lin(x) for lin in (self.lins)])

    def update(self, pred: torch.Tensor, labels: torch.Tensor, isTrain=True):
        loss_record = 0.0

        idx_start = 0
        for idx_end, loss_func, lin_optim in zip(self.end_idx_orders, self.loss_func_order, self.lin_optims):
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
    eff_optim = optim.Adam
    eff_lr = 1e-4
    lin_optims = [optim.Adam] * 7
    lin_lrs = [1e-4] * 7

    loss_func_order = [*[nn.CrossEntropyLoss()] * 6, nn.MSELoss()]

    bad_net = BadmintonNet(5, 5, loss_func_order).to('cuda:0')
    bad_net.init_optims(eff_optim=eff_optim, lin_optims=lin_optims, eff_lr=eff_lr, lin_lrs=lin_lrs)

    for _ in range(10):
        aa = bad_net(torch.randn((2, 5, 3, 512, 512)).to('cuda:0'))

        cc = torch.tensor([[*[0.0] * 5, 1.0, *[0.0, 1.0] * 4, *torch.rand(6), *[0.0] * 8, 1.0]] * 2).to('cuda:0')

        print(bad_net.update(aa, cc))
