import os, random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from submodules.UsefulTools.FileTools.FileOperator import get_filenames
from submodules.UsefulTools.FileTools.PickleOperator import load_pickle


class DatasetInfo:
    data_dir = Path('Data/Paper_used/pre_process/frame13')
    hit_dir_name = 'hit'
    miss_dir_name = 'miss'

    def __init__(self, data_id: str = None, data_dir: Path = Path('Data/Paper_used/pre_process/frame13')) -> None:
        self.data_dir = data_dir

        if data_id is not None:
            self.id = data_id
            self.data_dir = data_dir / data_id

        self.hit_dir = self.data_dir / self.hit_dir_name
        self.miss_dir = self.data_dir / self.miss_dir_name

        self.hit_pickle_paths = sorted(get_filenames(str(self.data_dir), f'{self.hit_dir_name}/*.pickle', withDirPath=False))
        self.miss_pickle_paths = sorted(get_filenames(str(self.data_dir), f'{self.miss_dir_name}/*.pickle', withDirPath=False))

    def show_datasets_size(self):
        print(f"hit: {len(self.hit_pickle_paths)}")
        print(f"miss: {len(self.miss_pickle_paths)}")


class CSVColumnNames:
    ShotSeq = 'ShotSeq'
    HitFrame = 'HitFrame'
    Hitter = 'Hitter'
    RoundHead = 'RoundHead'
    Backhand = 'Backhand'
    BallHeight = 'BallHeight'
    LandingX = 'LandingX'
    LandingY = 'LandingY'
    HitterLocationX = 'HitterLocationX'
    HitterLocationY = 'HitterLocationY'
    DefenderLocationX = 'DefenderLocationX'
    DefenderLocationY = 'DefenderLocationY'
    BallType = 'BallType'
    Winner = 'Winner'


class Frame13Dataset(Dataset):
    def __init__(self, side_range: int, dataset_info: DatasetInfo, frame_size: Tuple[int] = (720, 1280), isTrain=True) -> None:
        super(Frame13Dataset, self).__init__()

        self.side_range = side_range
        self.dataset_info = dataset_info
        self.frame_size = frame_size
        self.isTrain = isTrain

        self.center_idx = 6  # the center idx in 13 frames -> [0, ..., 12]
        self.num_frame = self.side_range * 2 + 1
        self.start_choice_idx = (3 - self.side_range) * 2
        self.end_choice_idx = self.start_choice_idx + self.num_frame
        self.choice_range = range(self.start_choice_idx, self.end_choice_idx)

        self.dataset_paths = (
            [*self.dataset_info.hit_pickle_paths, *self.dataset_info.miss_pickle_paths]
            if self.isTrain
            else self.dataset_info.hit_pickle_paths
        )
        self.len_dataset_path = len(self.dataset_paths)
        self.len_hit_data = len(self.dataset_info.hit_pickle_paths)

    def __getitem__(self, idx):
        data: torch.Tensor
        label: torch.Tensor
        data, label = load_pickle(str(self.dataset_info.data_dir / self.dataset_paths[idx]))

        start_idx = random.choice(self.choice_range)

        isHitData = self.len_hit_data // (idx + 1)
        isHitData = int(isHitData // (isHitData - 0.000001))
        hitFrame_label_idx = isHitData * (self.center_idx - start_idx) + (1 - isHitData) * (self.side_range * 2 + 1)

        data = data[start_idx : start_idx + self.num_frame]

        label_hitFrame = torch.zeros(self.num_frame + 1, dtype=torch.float32)
        label_hitFrame[hitFrame_label_idx] = 1.0

        label[-6::2] /= self.frame_size[1]
        label[-7::2] /= self.frame_size[0]

        label = torch.concat([label_hitFrame, label])

        return data, label, torch.tensor(hitFrame_label_idx, dtype=torch.int8), torch.tensor(isHitData, dtype=torch.bool)

    def __len__(self):
        return self.len_dataset_path


def get_dataloader(
    train_dir: Path,
    val_dir: Path,
    train_miss_rate: float = 0.2,
    side_range: int = 2,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = False,
    isDistributed=False,
):
    train_info = DatasetInfo(data_dir=train_dir)
    val_info = DatasetInfo(data_dir=val_dir)

    if train_info.miss_pickle_paths != []:
        num_train_miss = int(len(train_info.hit_pickle_paths) * train_miss_rate)
        miss_gap = num_train_miss - len(train_info.hit_pickle_paths)
        if miss_gap > 0:
            train_info.miss_pickle_paths.extend(random.sample(train_info.miss_pickle_paths, k=miss_gap))
        train_info.miss_pickle_paths = random.sample(train_info.miss_pickle_paths, k=num_train_miss)

    print("Number of train datasets:")
    train_info.show_datasets_size()

    print("\nNumber of val datasets:")
    val_info.show_datasets_size()

    train_data = Frame13Dataset(side_range, dataset_info=train_info, isTrain=True)
    val_data = Frame13Dataset(side_range, dataset_info=val_info)

    if isDistributed:
        train_loader = DataLoader(
            train_data,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=DistributedSampler(train_data),
        )
        val_loader = DataLoader(
            val_data,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=DistributedSampler(val_data),
        )
    else:
        train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_data, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


if __name__ == '__main__':
    # DatasetInfo().show_datasets_size()

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # trainInfo = DatasetInfo(data_dir=DatasetInfo.data_dir / 'train')

    train_dir = DatasetInfo.data_dir / 'train'
    train_dir_ids = os.listdir(train_dir)

    train_dataset = Frame13Dataset(side_range=2, dataset_info=DatasetInfo(data_dir=train_dir))

    for i in range(0, 6001, 2000):
        data, label, hit_idx, isHitData = train_dataset[i]

    val_dir = DatasetInfo.data_dir / 'val'
    val_dir_ids = os.listdir(val_dir)

    val_dataset = Frame13Dataset(side_range=2, dataset_info=DatasetInfo(data_dir=val_dir))

    for i in range(0, 201, 600):
        data, label, hit_idx, isHitData = val_dataset[i]

    train_loader, val_loader = get_dataloader(side_range=2, batch_size=2, num_workers=32, pin_memory=True)

    for data, label, hit_idx, isHitData in train_loader:
        print(hit_idx.shape)
