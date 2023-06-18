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
    sub_model_order_names = ['HitFrame', 'Hitter', 'RoundHead', 'Backhand', 'BallHeight', 'BallType', 'XY_Reg']
    end_idx_orders = [-23, -21, -19, -17, -15, -6, None]

    def __init__(self, in_seq: int):
        super(BadmintonNet, self).__init__()

        eff_out = 2048
        self.eff = EffNet(in_seq=in_seq, output_classes=eff_out)
        self.lins = nn.ModuleList(
            [
                LinNet(eff_out, in_seq + 1, isOneHot=True),  # 0~num_frame
                LinNet(eff_out, 2, isOneHot=True),  # -23~-21
                LinNet(eff_out, 2, isOneHot=True),  # -21~-19
                LinNet(eff_out, 2, isOneHot=True),  # -19~-17
                LinNet(eff_out, 2, isOneHot=True),  # -17~-15
                LinNet(eff_out, 9, isOneHot=True),  # -15~-6
                LinNet(eff_out, 6, isOneHot=False),  # -6~None
            ]
        )

        self.eff_optim: optim.Optimizer
        self.lin_optims: List[optim.Optimizer]

    def forward(self, x):
        x = self.eff(x)
        return torch.hstack([lin(x) for lin in (self.lins)])


class BadmintonNetOperator(nn.Module):
    sub_model_order_names = BadmintonNet.sub_model_order_names
    end_idx_orders = BadmintonNet.end_idx_orders

    def __init__(
        self,
        model: BadmintonNet,
        loss_func_order: List[nn.Module],
        eff_optim: optim.Optimizer,
        lin_optims: List[optim.Optimizer],
        eff_lr: float,
        lin_lrs: List[float],
        **kwargs,
    ):
        super(BadmintonNetOperator, self).__init__()

        self.loss_func_order = loss_func_order

        self.eff_optim: optim.Optimizer = eff_optim(model.eff.parameters(), lr=eff_lr, **kwargs)
        self.lin_optims: List[optim.Optimizer] = [
            optim(lin.parameters(), lr=lr, **kwargs) for optim, lr, lin in zip(lin_optims, lin_lrs, model.lins)
        ]

    def update(self, pred: torch.Tensor, labels: torch.Tensor, isTrain=True):
        loss_record = torch.zeros(8, dtype=torch.float32, requires_grad=False, device=pred.device)

        idx_start = 0
        for i, (idx_end, loss_func, lin_optim) in enumerate(zip(self.end_idx_orders, self.loss_func_order, self.lin_optims)):
            if isTrain:
                loss: torch.Tensor = loss_func(pred[:, idx_start:idx_end], labels[:, idx_start:idx_end])
                loss.backward(retain_graph=(i + 1) % len(self.sub_model_order_names))
                lin_optim.step()
                lin_optim.zero_grad()
            else:
                with torch.no_grad():
                    loss: torch.Tensor = loss_func(pred[:, idx_start:idx_end], labels[:, idx_start:idx_end])

            loss_record[i] = loss
            idx_start = idx_end

        loss_record[-1] = loss_record.sum()

        if isTrain:
            self.eff_optim.step()
            self.eff_optim.zero_grad()

        return loss_record

    def save(self, model: BadmintonNet, path: str, isFull: bool = False):
        if isFull:
            torch.save(model, path)
            torch.save(self, f'{path}_BadmintonNetUpdate.pickle')
        else:
            torch.save(model.state_dict(), path)
            torch.save(self.state_dict(), f'{path}_BadmintonNetUpdate.pickle')

    @staticmethod
    def load(obj: nn.Module, path: str, isFull: bool = False, device: str = 'cpu'):
        if isFull:
            obj = torch.load(path)
        else:
            obj.load_state_dict(torch.load(path, map_location=device))
            # obj.load_state_dict(torch.load(path))
        return obj


if __name__ == '__main__':
    # eff_optim = optim.Adam
    # eff_lr = 1e-4
    # lin_optims = [optim.Adam] * 7
    # lin_lrs = [1e-4] * 7

    # loss_func_order = [*[nn.CrossEntropyLoss()] * 6, nn.MSELoss()]

    # bad_net = BadmintonNet(5).to('cuda:1')
    # bad_net_operator = BadmintonNetOperator(
    #     bad_net,
    #     loss_func_order=loss_func_order,
    #     eff_optim=eff_optim,
    #     lin_optims=lin_optims,
    #     eff_lr=eff_lr,
    #     lin_lrs=lin_lrs,
    # )

    # for _ in range(10):
    #     aa = bad_net(torch.randn((2, 5, 3, 512, 512)).to('cuda:1'))

    #     cc = torch.tensor([[*[0.0] * 5, 1.0, *[0.0, 1.0] * 4, *torch.rand(6), *[0.0] * 8, 1.0]] * 2).to('cuda:1')

    #     print(bad_net_operator.update(aa, cc))

    eff_optim = optim.Adam
    eff_lr = 1e-4
    lin_optims = [optim.Adam] * 7
    lin_lrs = [1e-4] * 7
    loss_func_order = [*[nn.CrossEntropyLoss()] * 6, nn.MSELoss()]
    optim_kwargs = {}  # {'momentum': 0.9} # or {}
    in_seq = 5

    model_path = 'out/0612-2336_BadmintonNet_BS-15_Adam1.00e-04_Side2/bestAcc-Mean.pt'
    optmiz_path = (
        'out/0612-2336_BadmintonNet_BS-15_Adam1.00e-04_Side2/e097_lossSum-4.668e+00_accMean-0.000.pt_BadmintonNetUpdate.pickle'
    )

    net = BadmintonNet(in_seq=5)
    net = BadmintonNetOperator.load(net, model_path)

    bad_net_operator = BadmintonNetOperator(
        net,
        loss_func_order=loss_func_order,
        eff_optim=eff_optim,
        lin_optims=lin_optims,
        eff_lr=eff_lr,
        lin_lrs=lin_lrs,
        **optim_kwargs,
    )
    operator = BadmintonNetOperator.load(bad_net_operator, optmiz_path)
    aa = 0
