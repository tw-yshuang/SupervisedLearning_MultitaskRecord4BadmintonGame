import os, random
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset, DataLoader

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

        self.hit_frame13_ls: List[int] = list(map(int, [f.split('/')[-1].split('.pickle')[0] for f in self.hit_pickle_paths]))
        self.miss_frame13_ls: List[int] = list(map(int, [f.split('/')[-1].split('.pickle')[0] for f in self.miss_pickle_paths]))

    def show_datasets_size(self):
        print(f"hit: {len(self.hit_frame13_ls)}")
        print(f"miss: {len(self.miss_frame13_ls)}")


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
    def __init__(self, side_range: int, dataset_info: DatasetInfo, isTrain=True) -> None:
        super(Frame13Dataset, self).__init__()

        self.side_range = side_range
        self.dataset_info = dataset_info
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
        self.len_hit_data = len(self.dataset_info.hit_frame13_ls)

    def __getitem__(self, idx):
        data: torch.Tensor
        label: torch.Tensor
        data, label = load_pickle(str(self.dataset_info.data_dir / self.dataset_paths[idx]))

        start_idx = random.choice(self.choice_range)
        # hit_idx = self.center_idx - start_idx

        data = data[start_idx : start_idx + self.num_frame]

        label_hitFrame = torch.zeros(self.num_frame, dtype=torch.float32)
        isHitData = (self.len_hit_data - 1) // idx + 0.000001
        label_hitFrame[self.center_idx - start_idx] = isHitData / isHitData

        label = torch.concat([label_hitFrame, label])

        return data, label

    def __len__(self):
        return len(self.dataset_info)


def get_dataloader(side_range: int = 2, batch_size: int = 32, num_workers: int = 8, pin_memory: bool = False):
    train_data = Frame13Dataset(side_range, dataset_info=DatasetInfo(data_dir=DatasetInfo.data_dir / 'train'), isTrain=True)
    val_data = Frame13Dataset(side_range, dataset_info=DatasetInfo(data_dir=DatasetInfo.data_dir / 'val'))

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

    val_dir = DatasetInfo.data_dir / 'val'
    val_dir_ids = os.listdir(val_dir)

    val_dataset = Frame13Dataset(side_range=2, dataset_info=DatasetInfo(data_dir=val_dir))

    data, label = val_dataset[500]
