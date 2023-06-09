import random
from pathlib import Path
from typing import Callable, List, Union


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from submodules.UsefulTools.FileTools.WordOperator import str_format
from submodules.UsefulTools.FileTools.PickleOperator import load_pickle
from src.net.net import BadmintonNet
from src.transforms import IterativeCustomCompose
from src.accuracy import calculate, model_acc_names


# TODO: gogo
class ModelPerform:
    ...


class DL_Model:
    def __init__(
        self,
        model: Union[nn.Module, BadmintonNet],
        train_transforms: IterativeCustomCompose,
        test_transforms: IterativeCustomCompose,
        num_epoch: int,
        early_stop: int = 50,
        device: str = 'cuda',
        model_perform=None,
        acc_func: Callable = None,
    ) -> None:
        pass

        self.model = model
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.num_epoch = num_epoch
        self.device = device
        self.model_perform = model_perform
        self.acc_func = calculate if acc_func is None else acc_func
        self.early_stop = early_stop

        self.console = Console()

        self.best_loss_record = torch.zeros(8, dtype=torch.float32)
        self.best_acc_record = torch.zeros(13, dtype=torch.float32)

    def create_measure_table(self):
        loss_table = Table(show_header=True, header_style='bold magenta')
        acc_table = Table(show_header=True, header_style='bold magenta')

        loss_table.add_column("Loss", style="dim")
        [loss_table.add_column(name, justify='right') for name in [*self.model.sub_model_order_names, 'Sum']]

        acc_table.add_column("Acc", style="dim")
        [acc_table.add_column(name, justify='right') for name in [*model_acc_names, 'Mean']]

        return loss_table, acc_table

    def validating(self, loader: DataLoader):
        num_iter = 0
        self.model.eval()
        loss_record = torch.zeros_like(self.best_loss_record)
        acc_record = torch.zeros_like(self.best_acc_record)
        with torch.no_grad():
            for data, label, hit_idxs, isHits in tqdm(loader):
                data, label = data.to(self.device), label.to(self.device)

                batch_coordXYs = torch.stack(
                    [label[:, self.model.end_idx_orders[-2] :: 2], label[:, self.model.end_idx_orders[-2] + 1 :: 2]],
                ).permute(
                    1, 0, 2
                )  # stack like: [[relatedX, ...], [relatedY, ...]]

                data, batch_coordXYs = self.test_transforms(data, batch_coordXYs)
                batch_coordXYs = batch_coordXYs.permute(1, 0, 2)
                label[:, self.model.end_idx_orders[-2] :: 2] = batch_coordXYs[0]
                label[:, self.model.end_idx_orders[-2] + 1 :: 2] = batch_coordXYs[1]

                pred = self.model(data)
                loss_record[:] += self.model.update(pred, label, isTrain=False).cpu()
                acc_record[:] += self.acc_func(pred, label, hit_idxs, isHits).cpu()
                num_iter += 1

        loss_record /= num_iter
        acc_record /= num_iter

        return loss_record, acc_record

    def training(self, num_epoch: int, loader: DataLoader, val_loader: DataLoader = None, early_stop: int = 50, *args, **kwargs):
        data: torch.Tensor
        label: torch.Tensor
        hit_idxs: torch.Tensor
        isHits: torch.Tensor

        loss_records = torch.zeros((num_epoch, self.best_loss_record.shape[-1]), dtype=torch.float32)
        acc_records = torch.zeros((num_epoch, self.best_acc_record.shape[-1]), dtype=torch.float32)
        if val_loader is not None:
            val_loss_records = torch.zeros_like(loss_records)
            val_acc_records = torch.zeros_like(acc_records)
        for i in range(1, num_epoch + 1):
            loss_table, acc_table = self.create_measure_table()

            num_iter = 0
            self.model.train()
            for data, label, hit_idxs, isHits in tqdm(loader):
                # data, label = data.to(self.device), label.to(self.device)

                with torch.no_grad():
                    batch_coordXYs = torch.stack(
                        [label[:, self.model.end_idx_orders[-2] :: 2], label[:, self.model.end_idx_orders[-2] + 1 :: 2]],
                    ).permute(
                        1, 0, 2
                    )  # stack like: [[relatedX, ...], [relatedY, ...]]

                    data, batch_coordXYs = self.train_transforms(data, batch_coordXYs)
                    batch_coordXYs = batch_coordXYs.permute(1, 0, 2)
                    label[:, self.model.end_idx_orders[-2] :: 2] = batch_coordXYs[0]
                    label[:, self.model.end_idx_orders[-2] + 1 :: 2] = batch_coordXYs[1]

                data, label = data.to(self.device), label.to(self.device)
                pred = self.model(data)
                loss_records[i] += self.model.update(pred, label).cpu()
                acc_records[i] += self.acc_func(pred, label, hit_idxs, isHits).cpu()
                num_iter += 1

            loss_records[i] /= num_iter
            acc_records[i] /= num_iter

            loss_table.add_row('Train', *[f'{l:.3e}' for l in loss_records[i]])
            acc_table.add_row('Train', *[f'{a:.3f}' for a in acc_records[i]])

            if val_loader is not None:
                val_loss_records[i], val_acc_records[i] = self.validating(val_loader)

                loss_table.add_row('val', *[f'{l:.3e}' for l in val_loss_records[i]])
                acc_table.add_row('val', *[f'{a:.3f}' for a in val_acc_records[i]])

            self.console.print(loss_table)
            self.console.print(acc_table)


if __name__ == '__main__':
    from torch import optim
    from torchvision import transforms

    from src.data_process import get_dataloader, DatasetInfo
    from src.transforms import RandomCrop, RandomResizedCrop, RandomHorizontalFlip, RandomRotation

    eff_optim = optim.Adam
    eff_lr = 1e-4
    lin_optims = [optim.Adam] * 7
    lin_lrs = [1e-4] * 7

    loss_func_order = [*[nn.CrossEntropyLoss()] * 6, nn.MSELoss()]

    sizeHW = (512, 512)
    argumentation_order_ls = [
        RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
        RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur([3, 3]),
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, hue=0.2, contrast=0.5, saturation=0.2)], p=0.75),
        # transforms.RandomPosterize(6, p=0.15),
        # transforms.RandomEqualize(p=0.15),
        # transforms.RandomSolarize(128, p=0.1),
        # transforms.RandomInvert(p=0.05),
        transforms.RandomApply(
            [transforms.ElasticTransform(alpha=random.random() * 200.0, sigma=8.0 + random.random() * 7.0)], p=0.25
        ),
        RandomRotation(degrees=[-5, 5], p=0.75),
    ]

    train_iter_compose = IterativeCustomCompose(
        [
            *argumentation_order_ls,
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ],
        transform_img_size=sizeHW,
    )
    test_iter_compose = IterativeCustomCompose(
        [
            transforms.Resize(sizeHW),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ],
        transform_img_size=sizeHW,
    )

    bad_net = BadmintonNet(5, 5, loss_func_order).to('cuda')
    bad_net.init_optims(eff_optim=eff_optim, lin_optims=lin_optims, eff_lr=eff_lr, lin_lrs=lin_lrs)

    train_set, val_set = get_dataloader(
        train_dir=DatasetInfo.data_dir / 'train',
        val_dir=DatasetInfo.data_dir / 'val',
        batch_size=16,
        num_workers=16,
        pin_memory=True,
    )

    model_process = DL_Model(bad_net, train_iter_compose, test_iter_compose, 10, device='cuda')
    model_process.training(10, train_set, val_set)
