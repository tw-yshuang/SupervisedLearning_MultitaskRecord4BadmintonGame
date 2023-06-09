from typing import List

import torch

idx_Hitter_start = -23
idx_LandingX_start = -6
haft_size = torch.tensor([720, 1280]) / 2


def calculate(preds: torch.Tensor, labels: torch.Tensor, hit_idxs: torch.Tensor, isHits: torch.Tensor):
    with torch.no_grad():
        hit_preds, hit_labels, hit_idxs = preds[isHits], labels[isHits], hit_idxs[isHits]

        cls_idx_select = hit_labels[:, :idx_LandingX_start].type(torch.bool)
        cls_acc_tensor = hit_preds[:, :idx_LandingX_start][cls_idx_select].reshape(-1, 6).mean(dim=0)

        reg_acc_tensor = 1 - torch.square(hit_labels[:, idx_LandingX_start:] - hit_preds[:, idx_LandingX_start:]).mean(dim=0)

        return torch.hstack([cls_acc_tensor, reg_acc_tensor])


if __name__ == '__main__':
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torch.optim as optim

    PROJECT_DIR = Path(__file__).resolve().parents[1]
    if __name__ == '__main__':
        import sys

        sys.path.append(str(PROJECT_DIR))

    from src.net.net import BadmintonNet

    eff_optim = optim.Adam
    eff_lr = 1e-4
    lin_optims = [optim.Adam] * 7
    lin_lrs = [1e-4] * 7

    loss_func_order = [*[nn.CrossEntropyLoss()] * 6, nn.MSELoss()]

    bad_net = BadmintonNet(5, 5, loss_func_order).to('cuda:0')
    bad_net.init_optims(eff_optim=eff_optim, lin_optims=lin_optims, eff_lr=eff_lr, lin_lrs=lin_lrs)

    for _ in range(10):
        aa = bad_net(torch.randn((3, 5, 3, 512, 512)).to('cuda:0'))

        cc = torch.tensor([[*[0.0] * 5, 1.0, *[0.0, 1.0] * 4, *[0.0] * 8, 1.0, *torch.rand(6)]] * 3).to('cuda:0')
        hit_idxs = torch.tensor([6] * 3, dtype=torch.int8)
        isHits = torch.tensor([0, 1, 1], dtype=torch.bool)
        print(calculate(aa, cc, hit_idxs, isHits))
