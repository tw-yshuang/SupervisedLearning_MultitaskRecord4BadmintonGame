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
from src.net.net import BadmintonNet, BadmintonNetOperator
from src.transforms import IterativeCustomCompose
from src.evaluate.accuracy import calculate, model_acc_names


class ModelPerform:
    def __init__(
        self,
        loss_order_names: List[str],
        acc_order_names: List[str],
        loss_records: torch.Tensor,
        acc_records: torch.Tensor,
        test_loss_records: torch.Tensor = None,
        test_acc_records: torch.Tensor = None,
    ) -> None:
        self.acc_df = pd.DataFrame(self.convert(acc_records, acc_order_names))
        self.loss_df = pd.DataFrame(self.convert(loss_records, loss_order_names))

        self.haveTestRecords = test_acc_records is not None
        if self.haveTestRecords:
            self.test_loss_df = pd.DataFrame(self.convert(test_loss_records, loss_order_names))
            self.test_acc_df = pd.DataFrame(self.convert(test_acc_records, acc_order_names))

    @staticmethod
    def convert(records: torch.Tensor, order_names: List[str]):
        return {name: records[:, i].tolist() for i, name in enumerate(order_names)}

    def save(self, saveDir: str, start_row: int = 0, end_row: int = None):
        self.loss_df[start_row:end_row].to_csv(f'{saveDir}/train_loss.csv')
        self.acc_df[start_row:end_row].to_csv(f'{saveDir}/train_acc.csv')

        if self.haveTestRecords:
            self.test_loss_df[start_row:end_row].to_csv(f'{saveDir}/val_loss.csv')
            self.test_acc_df[start_row:end_row].to_csv(f'{saveDir}/val_acc.csv')


class DL_Model:
    def __init__(
        self,
        model: Union[nn.Module, BadmintonNet],
        model_operator: BadmintonNetOperator,
        train_transforms: IterativeCustomCompose,
        test_transforms: IterativeCustomCompose,
        device: str = 'cuda',
        model_perform=None,
        acc_func: Callable = None,
    ) -> None:
        self.model = model
        self.model_operator = model_operator
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.device = device
        self.model_perform = model_perform
        self.acc_func = calculate if acc_func is None else acc_func

        self.console = Console()
        self.loss_order_names = [*self.model.sub_model_order_names, 'Sum']
        self.acc_order_names = model_acc_names

        self.epoch = 0
        self.best_epoch = 0
        self.best_loss_record = torch.ones(len(self.loss_order_names), dtype=torch.float32) * 100
        self.best_acc_record = torch.ones(len(self.acc_order_names), dtype=torch.float32) * 0

    def create_measure_table(self):
        loss_table = Table(show_header=True, header_style='bold magenta')
        acc_table = Table(show_header=True, header_style='bold magenta')

        loss_table.add_column("Loss", style="dim")
        [loss_table.add_column(name, justify='right') for name in self.loss_order_names]

        acc_table.add_column("Acc", style="dim")
        [acc_table.add_column(name, justify='right') for name in self.acc_order_names]

        return loss_table, acc_table

    def validating(self, loader: DataLoader):
        num_iter = 0
        self.model.eval()
        loss_record = torch.zeros_like(self.best_loss_record)
        acc_record = torch.zeros_like(self.best_acc_record)
        with torch.no_grad():
            for data, label, hit_idxs, isHits, _ in tqdm(loader):
                data, label = data.to(self.device), label.to(self.device)

                batch_coordXYs = torch.stack(
                    [label[:, self.model_operator.end_idx_orders[-2] :: 2], label[:, self.model_operator.end_idx_orders[-2] + 1 :: 2]],
                ).permute(
                    1, 0, 2
                )  # stack like: [[relatedX, ...], [relatedY, ...]]

                data, batch_coordXYs = self.test_transforms(data, batch_coordXYs)
                batch_coordXYs = batch_coordXYs.permute(1, 0, 2)
                label[:, self.model_operator.end_idx_orders[-2] :: 2] = batch_coordXYs[0]
                label[:, self.model_operator.end_idx_orders[-2] + 1 :: 2] = batch_coordXYs[1]

                pred = self.model(data)
                loss_record[:] += self.model_operator.update(pred, label, isTrain=False).cpu()
                acc_record[:] += self.acc_func(pred, label, hit_idxs, isHits).cpu()
                num_iter += 1

        loss_record /= num_iter
        acc_record /= num_iter

        return loss_record, acc_record

    def training(
        self,
        num_epoch: int,
        loader: DataLoader,
        val_loader: DataLoader = None,
        saveDir: Path = PROJECT_DIR,
        early_stop: int = 50,
        checkpoint: int = 20,
        *args,
        **kwargs,
    ):
        data: torch.Tensor
        label: torch.Tensor
        hit_idxs: torch.Tensor
        isHits: torch.Tensor

        loss_records = torch.zeros((num_epoch, self.best_loss_record.shape[-1]), dtype=torch.float32)
        acc_records = torch.zeros((num_epoch, self.best_acc_record.shape[-1]), dtype=torch.float32)
        if val_loader is not None:
            val_loss_records = torch.zeros_like(loss_records)
            val_acc_records = torch.zeros_like(acc_records)

        isStop = False
        for self.epoch in range(num_epoch):
            loss_table, acc_table = self.create_measure_table()

            isBest = False
            num_iter = 0
            num_missM_nan = 0
            self.model.train()
            for data, label, hit_idxs, isHits, _ in tqdm(loader):
                data, label = data.to(self.device), label.to(self.device)

                with torch.no_grad():
                    batch_coordXYs = torch.stack(
                        [
                            label[:, self.model_operator.end_idx_orders[-2] :: 2],
                            label[:, self.model_operator.end_idx_orders[-2] + 1 :: 2],
                        ],
                    ).permute(
                        1, 0, 2
                    )  # stack like: [[relatedX, ...], [relatedY, ...]]

                    data, batch_coordXYs = self.train_transforms(data, batch_coordXYs)
                    batch_coordXYs = batch_coordXYs.permute(1, 0, 2)
                    label[:, self.model_operator.end_idx_orders[-2] :: 2] = batch_coordXYs[0]
                    label[:, self.model_operator.end_idx_orders[-2] + 1 :: 2] = batch_coordXYs[1]

                # data, label = data.to(self.device), label.to(self.device)
                pred = self.model(data)
                loss_records[self.epoch] += self.model_operator.update(pred, label).cpu()
                a_r = self.acc_func(pred, label, hit_idxs, isHits).cpu()
                num_missM_nan += 1 - (a_r[-1] // (a_r[-1] - 0.0000001))
                acc_records[self.epoch] += a_r
                num_iter += 1

            loss_records[self.epoch] /= num_iter
            acc_records[self.epoch, :-2] /= num_iter
            acc_records[self.epoch, -2:] /= num_iter - num_missM_nan

            loss_table.add_row('Train', *[f'{l:.3e}' for l in loss_records[self.epoch]])
            acc_table.add_row('Train', *[f'{a:.3f}' for a in acc_records[self.epoch]])

            if val_loader is not None:
                val_loss_records[self.epoch], val_acc_records[self.epoch] = self.validating(val_loader)

                loss_table.add_row('val', *[f'{l:.3e}' for l in val_loss_records[self.epoch]])
                acc_table.add_row('val', *[f'{a:.3f}' for a in val_acc_records[self.epoch]])

                best_loss_checker = self.best_loss_record > val_loss_records[self.epoch]
                self.best_loss_record[best_loss_checker] = val_loss_records[self.epoch, best_loss_checker]

                best_acc_checker = self.best_acc_record < val_acc_records[self.epoch]
                self.best_acc_record[best_acc_checker] = val_acc_records[self.epoch, best_acc_checker]

                if best_acc_checker.any() or best_loss_checker.any():
                    self.best_epoch = self.epoch
                    isBest = True

            self.console.print(loss_table)
            self.console.print(acc_table)

            # * Save Stage
            isCheckpoint = self.epoch % checkpoint == 0
            if self.best_epoch:
                save_path = f'lossSum-{val_loss_records[self.epoch, -1]:.3e}_accMean-{val_acc_records[self.epoch, -6]:.3f}.pt'
                isStop = early_stop == (self.epoch - self.best_epoch)
            else:
                save_path = f'lossSum-{loss_records[self.epoch, -1]:.3e}_accMean-{acc_records[self.epoch, -6]:.3f}.pt'

            save_path_heads: List[str] = []
            if isCheckpoint:
                save_path_heads.append(f'checkpoint_e{self.epoch:03}')
            if isBest:
                save_path_heads.extend(
                    [f'bestLoss-{name}' for name, is_best in zip(self.loss_order_names, best_loss_checker) if is_best],
                )
                save_path_heads.extend(
                    [f'bestAcc-{name}' for name, is_best in zip(self.acc_order_names, best_acc_checker) if is_best],
                )

            isStop += self.epoch + 1 == num_epoch
            if isStop:
                save_path_heads.append(f'final_e{self.epoch:03}_')

            for i, path_head in enumerate(save_path_heads):
                if i == 0:
                    epoch_path = f'e{self.epoch:03}_{save_path}'
                    self.model_operator.save(self.model, str(saveDir / epoch_path))
                    print(f"Save Model: {str_format(str(epoch_path), fore='g')}")
                    model_perform = ModelPerform(
                        self.loss_order_names,
                        self.acc_order_names,
                        loss_records[: self.epoch + 1],
                        acc_records[: self.epoch + 1],
                        val_loss_records[: self.epoch + 1],
                        val_acc_records[: self.epoch + 1],
                    )
                    model_perform.save(str(saveDir))

                path: Path = saveDir / f'{path_head}.pt'
                path.unlink(missing_ok=True)
                path.symlink_to(epoch_path)
                print(f"symlink: {str_format(str(path_head), fore='y'):<36} -> {epoch_path}")

            if isStop:
                print(str_format("Stop!!", fore='y'))
                break

        if val_loader is None:
            return loss_records, acc_records
        return loss_records, acc_records, val_loss_records, val_acc_records


if __name__ == '__main__':
    import time

    from torch import optim
    from torchvision import transforms

    from src.data_process import get_dataloader, DatasetInfo
    from src.transforms import RandomCrop, RandomResizedCrop, RandomHorizontalFlip, RandomRotation

    from submodules.UsefulTools.FileTools.FileOperator import check2create_dir
    from submodules.UsefulTools.FileTools.PickleOperator import save_pickle

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

    eff_optim = optim.Adam
    eff_lr = 1e-4
    lin_optims = [optim.Adam] * 7
    lin_lrs = [1e-4] * 7
    loss_func_order = [*[nn.CrossEntropyLoss()] * 6, nn.MSELoss()]

    bad_net = BadmintonNet(5, 5, loss_func_order).to('cuda')
    bad_net.init_optims(eff_optim=eff_optim, lin_optims=lin_optims, eff_lr=eff_lr, lin_lrs=lin_lrs)

    BATCH_SIZE = 32

    train_set, val_set = get_dataloader(
        train_dir=DatasetInfo.data_dir / 'test',
        val_dir=DatasetInfo.data_dir / 'test',
        batch_size=BATCH_SIZE,
        num_workers=32,
        pin_memory=True,
    )

    saveDir = f'out/{time.strftime("%m%d-%H%M")}_{bad_net.__class__.__name__}_BS-{BATCH_SIZE}'
    check2create_dir(saveDir)

    model_process = DL_Model(bad_net, train_iter_compose, test_iter_compose, device='cuda')
    records_tuple = model_process.training(3, train_set, val_set, saveDir=Path(saveDir), early_stop=3, checkpoint=3)

    for records, name in zip(records_tuple, ('train_loss_records', 'train_acc_records', 'val_loss_records', 'val_acc_records')):
        save_pickle(records, f'{saveDir}/{name}.pickle')

    model_perform = ModelPerform(model_process.loss_order_names, model_process.acc_order_names, *records_tuple)
    model_perform.loss_df.to_csv(f'{saveDir}/train_loss.csv')
    model_perform.acc_df.to_csv(f'{saveDir}/train_acc.csv')
    model_perform.test_loss_df.to_csv(f'{saveDir}/val_loss.csv')
    model_perform.test_acc_df.to_csv(f'{saveDir}/val_acc.csv')
