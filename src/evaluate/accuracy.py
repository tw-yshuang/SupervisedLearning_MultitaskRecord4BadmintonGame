from typing import List
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from src.net.net import BadmintonNet

# is count from behind(negative value)
num_cls_task = len(BadmintonNet.end_idx_orders) - 1
end_idx_orders = BadmintonNet.end_idx_orders
idx_Hitter_start = end_idx_orders[0]
idx_LandingX_start = end_idx_orders[-2]

model_acc_names = [
    'HitFrame',
    'Hitter',
    'RoundHead',
    'Backhand',
    'BallHeight',
    'BallType',
    'LandingX',
    'LandingY',
    'HitterLocationX',
    'HitterLocationY',
    'DefenderLocationX',
    'DefenderLocationY',
    'Mean',
    'HitFrameP',
    'MissH',
    'MissHP',
    'MissM',
    'MissMP',
]


def calculate(preds: torch.Tensor, labels: torch.Tensor, hit_idxs: torch.Tensor, isHits: torch.Tensor):
    with torch.no_grad():
        hit_preds, not_hit_preds, hit_labels, hitFrame_idxs_select = (
            preds[isHits],
            preds[~isHits],
            labels[isHits],
            hit_idxs[isHits].type(torch.int),
        )

        miss_idx = not_hit_preds.shape[-1] + idx_Hitter_start - 1  # because idx_Hitter_start is count from behind(negative value)

        # miss factor analysis
        missHP = hit_preds[:, miss_idx].mean()
        missMP = not_hit_preds[:, miss_idx].mean()

        not_hit_pred_idxs = not_hit_preds[:, :idx_Hitter_start].argmax(dim=1)
        missM = (not_hit_pred_idxs == miss_idx).sum() / not_hit_preds.shape[0]

        start_idx = 0
        cls_acc_tensor = torch.zeros(num_cls_task, dtype=torch.float32, device=preds.device)
        for i, end_idx in enumerate(end_idx_orders[:num_cls_task]):
            one_task_hit_pred_idxs = hit_preds[:, start_idx:end_idx].argmax(dim=1)
            one_task_label_pred_idxs = hit_labels[:, start_idx:end_idx].argmax(dim=1)
            cls_acc_tensor[i] = (one_task_hit_pred_idxs == one_task_label_pred_idxs).sum() / hit_preds.shape[0]
            start_idx = end_idx

            if i == 0:
                missH = (one_task_hit_pred_idxs == miss_idx).sum() / hit_preds.shape[0]
                hitFrameP = hit_preds[range(hitFrame_idxs_select.shape[0]), hitFrame_idxs_select].mean()

        # # TODO: replace to the argmax()
        # cls_idx_select = hit_labels[:, :idx_LandingX_start].type(torch.bool)
        # cls_acc_tensor = hit_preds[:, :idx_LandingX_start][cls_idx_select].reshape(-1, 6).mean(dim=0)

        reg_acc_tensor = 1 - torch.abs(hit_labels[:, idx_LandingX_start:] - hit_preds[:, idx_LandingX_start:]).mean(dim=0)

        if missMP.isnan() or missM.isnan():
            missMP, missM = torch.zeros_like(missMP), torch.zeros_like(missM)

        acc_record = torch.hstack(
            [
                cls_acc_tensor,
                reg_acc_tensor,
                torch.hstack([cls_acc_tensor, reg_acc_tensor]).mean(),
                hitFrameP,
                missH,
                missHP,
                missM,
                missMP,
            ]
        )

        return acc_record


if __name__ == '__main__':
    bad_net = BadmintonNet(5).to('cuda:2')

    for _ in range(10):
        aa = bad_net(torch.randn((3, 5, 3, 512, 512)).to('cuda:2'))

        cc = torch.tensor([[*[0.0] * 5, 1.0, *[0.0, 1.0] * 4, *[0.0] * 8, 1.0, *torch.rand(6)]] * 3).to('cuda:2')
        hit_idxs = torch.tensor([6] * 3, dtype=torch.int8)
        isHits = torch.tensor([1, 1, 1], dtype=torch.bool)
        print(calculate(aa, cc, hit_idxs, isHits))
