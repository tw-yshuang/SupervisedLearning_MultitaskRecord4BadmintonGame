import os, random, time
from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from src.data_process import get_dataloader, DatasetInfo
from src.transforms import IterativeCustomCompose, RandomCrop, RandomResizedCrop, RandomHorizontalFlip, RandomRotation
from src.net.net import BadmintonNet
from src.training import ModelPerform, DL_Model

# from src.net.data_parallel_my_v2 import BalancedDataParallel

from submodules.UsefulTools.FileTools.FileOperator import check2create_dir
from submodules.UsefulTools.FileTools.PickleOperator import save_pickle

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #! ========== Hyperparameter ==========

    # * Datasets
    BATCH_SIZE = 15
    NUM_WORKERS = 16
    side_range = 2
    train_miss_rate = 1 / (side_range * 2 + 1)

    # * model
    eff_optim = optim.SGD
    eff_lr = 1e-3
    lin_optims = [optim.SGD] * 7
    lin_lrs = [1e-3] * 7
    loss_func_order = [*[nn.CrossEntropyLoss()] * 6, nn.MSELoss()]
    optim_kwargs = {'momentum': 0.9}
    in_seq = side_range * 2 + 1

    # * Train Process
    NUM_EPOCHS = 30
    EARLY_STOP = 10
    CHECKPOINT = 5

    #! ========== Argumentation ==========
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
        device=device,
    )
    test_iter_compose = IterativeCustomCompose(
        [
            transforms.Resize(sizeHW),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ],
        transform_img_size=sizeHW,
        device=device,
    )

    #! ========== Datasets ==========

    train_loader, val_loader = get_dataloader(
        train_dir=DatasetInfo.data_dir / 'train',
        val_dir=DatasetInfo.data_dir / 'val',
        train_miss_rate=train_miss_rate,
        side_range=side_range,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    #! ========== Network ==========

    bad_net = BadmintonNet(in_seq, loss_func_order).to(device)
    bad_net.init_optims(eff_optim=eff_optim, lin_optims=lin_optims, eff_lr=eff_lr, lin_lrs=lin_lrs, **optim_kwargs)
    # parallel_model = DistributedDataParallel(BATCH_SIZE % 16, bad_net).to(device)
    # parallel_model = DistributedDataParallel(bad_net)
    # parallel_model.update = bad_net.update
    # parallel_model.save = bad_net.save
    # parallel_model.sub_model_order_names = bad_net.sub_model_order_names
    # parallel_model.end_idx_orders = bad_net.end_idx_orders
    # parallel_model.loss_func_order = bad_net.loss_func_order
    # parallel_model.eff_optim = bad_net.eff_optim
    # parallel_model.lin_optims = bad_net.lin_optims

    #! ========== Train Process ==========
    saveDir = f'out/{time.strftime("%m%d-%H%M")}_{bad_net.__class__.__name__}_BS-{BATCH_SIZE}_{eff_optim.__name__}{eff_lr:.2e}'
    check2create_dir(saveDir)

    # model_process = DL_Model(parallel_model, train_iter_compose, test_iter_compose, device=device)
    model_process = DL_Model(bad_net, train_iter_compose, test_iter_compose, device=device)
    records_tuple = model_process.training(
        NUM_WORKERS, train_loader, val_loader, saveDir=Path(saveDir), early_stop=EARLY_STOP, checkpoint=CHECKPOINT
    )

    #! ========== Records Saving ==========
    for records, name in zip(records_tuple, ('train_loss_records', 'train_acc_records', 'val_loss_records', 'val_acc_records')):
        save_pickle(records, f'{saveDir}/{name}.pickle')

    model_perform = ModelPerform(model_process.loss_order_names, model_process.acc_order_names, *records_tuple)
    model_perform.loss_df.to_csv(f'{saveDir}/train_loss.csv')
    model_perform.acc_df.to_csv(f'{saveDir}/train_acc.csv')
    model_perform.test_loss_df.to_csv(f'{saveDir}/val_loss.csv')
    model_perform.test_acc_df.to_csv(f'{saveDir}/val_acc.csv')
