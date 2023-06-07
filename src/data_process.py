import os
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from submodules.UsefulTools.FileTools.FileSearcher import get_filenames, check2create_dir
from submodules.UsefulTools.FileTools.PickleOperator import load_pickle, save_pickle
from src.transforms import CustomCompose, RandomHorizontalFlip, RandomResizedCrop, RandomRotation


class DatasetInfo:
    data_dir = Path('./Data/pre-process')
    dataset_pickle_dir = Path('./Data/dataset_pickle')
    label_dir = Path('./Data/part1/train')
    image_dir = Path('image')
    predict_dir = Path('predict')
    mask_dir = Path('mask')
    predict_merge_dir = Path('predict_merge')
    ball_mask5_dir = Path('ball_mask5_dir')

    def __init__(self, data_id: str, isTrain=True) -> None:
        self.id = data_id
        self.data_dir = DatasetInfo.data_dir / data_id
        self.image_dir = self.data_dir / DatasetInfo.image_dir
        self.mask_dir = self.data_dir / DatasetInfo.mask_dir
        self.predict_dir = self.data_dir / DatasetInfo.predict_dir
        self.predict_merge_dir = self.data_dir / DatasetInfo.predict_merge_dir
        self.ball_mask5_dir = self.data_dir / DatasetInfo.ball_mask5_dir
        self.frame5_start_ls: List[str] = [
            int(f.split('.pickle')[0]) for f in get_filenames(str(self.ball_mask5_dir), '*.pickle', withDirPath=False)
        ]

        if isTrain:
            self.label_csv = DatasetInfo.label_dir / data_id / f'{data_id}_S2.csv'
        else:
            self.dataset_pickle_dir = self.data_dir / str(self.dataset_pickle_dir).split('/')[-1]
            check2create_dir(self.dataset_pickle_dir)


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


def order_data(dataset_infos: List[DatasetInfo], len_dataset: int):
    data_order_ls = []
    data_id2startIdx_arr = np.zeros((len_dataset, 2), dtype=np.uint32)
    continue_idx = 0
    for i, dataset_info in enumerate(dataset_infos):
        data_id2startIdx_arr[i] = dataset_info.id, continue_idx
        data_order_ls.extend(dataset_info.frame5_start_ls)
        continue_idx += len(dataset_info.frame5_start_ls)

    data_order_arr = np.array(data_order_ls, dtype=np.uint32)

    return data_id2startIdx_arr, data_order_arr


def Processing(compose: Union[CustomCompose, transforms.Compose]):
    def __processing(imgs: List[np.ndarray], label_info: Union[pd.Series, int] = 0):
        process_imgs = torch.stack([torch.tensor(img.transpose((2, 0, 1))) for img in imgs]).type(torch.uint8)

        if not isinstance(label_info, int):  # hit_frame in it
            process_label = torch.zeros(32, dtype=torch.float32)
            process_label[label_info.at[CSVColumnNames.HitFrame]] = 1.0  # HitFrame: [0~5] one-hot
            process_label[6 if label_info.at[CSVColumnNames.Hitter] == 'A' else 7] = 1.0  # Hitter: [6,7] one-hot
            process_label[7 + label_info.at[CSVColumnNames.RoundHead]] = 1.0  # RoundHead: [8,9] one-hot
            process_label[9 + label_info.at[CSVColumnNames.Backhand]] = 1.0  # Backhand: [10,11] one-hot
            process_label[11 + label_info.at[CSVColumnNames.BallHeight]] = 1.0  # BallHeight: [12,13] one-hot
            process_label[14:20] = torch.from_numpy(
                label_info.loc[
                    [
                        CSVColumnNames.LandingX,  # LandingX: 14
                        CSVColumnNames.LandingY,  # LandingY: 15
                        CSVColumnNames.HitterLocationX,  # HitterLocationX: 16
                        CSVColumnNames.HitterLocationY,  # HitterLocationY: 17
                        CSVColumnNames.DefenderLocationX,  # DefenderLocationX: 18
                        CSVColumnNames.DefenderLocationY,  # DefenderLocationY: 19
                    ],
                ].to_numpy(dtype=np.float32)
            )
            process_label[19 + label_info.at[CSVColumnNames.BallType]] = 1.0  # BallType: [20~28] one-hot

            w_id = 29
            w = label_info.at[CSVColumnNames.Winner]
            if w == 'B':
                w_id += 1
            elif w == 'X':
                w_id += 2
            process_label[w_id] = 1.0  # Winner: [29~31] one-hot

            coordXYs = torch.stack([process_label[14:20:2], process_label[15:20:2]])  # stack like: [[relatedX, ...], [relatedY, ...]]

            process_imgs, coordXYs = compose(process_imgs, coordXYs)
            process_label[14:20:2] = coordXYs[0]
            process_label[15:20:2] = coordXYs[1]

            return process_imgs, process_label

        if label_info == -1:  # hit_frame miss
            process_label = torch.zeros(32, dtype=torch.float32)
            process_label[5] = 1.0
            process_imgs = compose(process_imgs)[0]
            return process_imgs, process_label

        return compose(process_imgs)[0]  # label_info == 0, test stage

    return __processing


def Processing2Tensor():
    def __processing2tensor(imgs: List[np.ndarray], label_info: Union[pd.Series, int] = 0):
        process_imgs = torch.stack([torch.tensor(img.transpose((2, 0, 1))) for img in imgs]).type(torch.uint8)

        if not isinstance(label_info, int):  # hit_frame in it
            process_label = torch.zeros(32, dtype=torch.float32)
            process_label[label_info.at[CSVColumnNames.HitFrame]] = 1.0  # HitFrame: [0~5] one-hot
            process_label[6 if label_info.at[CSVColumnNames.Hitter] == 'A' else 7] = 1.0  # Hitter: [6,7] one-hot
            process_label[7 + label_info.at[CSVColumnNames.RoundHead]] = 1.0  # RoundHead: [8,9] one-hot
            process_label[9 + label_info.at[CSVColumnNames.Backhand]] = 1.0  # Backhand: [10,11] one-hot
            process_label[11 + label_info.at[CSVColumnNames.BallHeight]] = 1.0  # BallHeight: [12,13] one-hot
            process_label[14:20] = torch.from_numpy(
                label_info.loc[
                    [
                        CSVColumnNames.LandingX,  # LandingX: 14
                        CSVColumnNames.LandingY,  # LandingY: 15
                        CSVColumnNames.HitterLocationX,  # HitterLocationX: 16
                        CSVColumnNames.HitterLocationY,  # HitterLocationY: 17
                        CSVColumnNames.DefenderLocationX,  # DefenderLocationX: 18
                        CSVColumnNames.DefenderLocationY,  # DefenderLocationY: 19
                    ],
                ].to_numpy(dtype=np.float32)
            )
            process_label[19 + label_info.at[CSVColumnNames.BallType]] = 1.0  # BallType: [20~28] one-hot

            w_id = 29
            w = label_info.at[CSVColumnNames.Winner]
            if w == 'B':
                w_id += 1
            elif w == 'X':
                w_id += 2
            process_label[w_id] = 1.0  # Winner: [29~31] one-hot

            return process_imgs, process_label

        if label_info == -1:  # hit_frame miss
            process_label = torch.zeros(32, dtype=torch.float32)
            process_label[5] = 1.0
            return process_imgs, process_label

        return process_imgs  # label_info == 0, test stage

    return __processing2tensor


class Img5Dataset(Dataset):
    def __init__(self, dataset_ids: List[str], processing: Processing, isTrain=True) -> None:
        super(Img5Dataset, self).__init__()

        self.processing = processing
        self.isTrain = isTrain

        dataset_infos = [DatasetInfo(dataset_id, self.isTrain) for dataset_id in dataset_ids]
        self.frameID2startIdx_arr, self.data_order_arr = order_data(dataset_infos, len(dataset_ids))

        if self.isTrain:
            self.label_csvs = [dataset_info.label_csv for dataset_info in dataset_infos]
        else:
            self.dataset_infos = dataset_infos

    def __getitem__(self, idx):
        frame5_start = self.data_order_arr[idx]
        data_id = np.where(self.frameID2startIdx_arr[:, 1] <= idx)[0][-1]
        data_dir = DatasetInfo.data_dir / f'{self.frameID2startIdx_arr[data_id][0]:05d}' / DatasetInfo.predict_merge_dir

        filenames = [str(data_dir / f'{i}.jpg') for i in range(frame5_start + 1, frame5_start + 6)]  # "*.jpg" start from 1
        imgs = [cv2.imread(filename) for filename in filenames]  # TODO: can change to use torchvision.io.read_image()
        imgs = [imgs[i - 1].copy() if img is None else img for i, img in enumerate(imgs)]

        if self.isTrain:
            df = pd.read_csv(str(self.label_csvs[data_id]))
            hit_frames = df.loc[:, CSVColumnNames.HitFrame].to_numpy()
            hit_idx = np.where((frame5_start <= hit_frames) & (hit_frames < (frame5_start + 5)))[0]
            if hit_idx.size == 0:
                return self.processing(imgs, label_info=-1)

            hit_idx = hit_idx[0]
            df.at[hit_idx, CSVColumnNames.HitFrame] -= frame5_start
            return self.processing(imgs, label_info=df.loc[hit_idx])
        else:
            self.data_dir = DatasetInfo(f'{self.frameID2startIdx_arr[data_id][0]:05d}', isTrain=False).dataset_pickle_dir
            self.frame5_start = frame5_start
            return self.processing(imgs, label_info=0)

    def __len__(self):
        return self.data_order_arr.shape[0]


class Pickle5Dataset(Dataset):
    def __init__(
        self, compose: Union[CustomCompose, transforms.Compose], pickle_dir: str, filenames: Union[List[str], None] = None
    ) -> None:
        super(Pickle5Dataset, self).__init__()

        self.compose = compose

        if isinstance(filenames, list):
            self.filenames = filenames
        else:
            self.filenames = get_filenames(pickle_dir, specific_name='*.pickle')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        return load_pickle(self.filenames[index])


def get_dataloader(
    preprocess_dir: str = None,
    filenames: Union[List[str], None] = None,
    dataset_rate=0.8,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = False,
    usePickle: bool = False,
    compose: Union[CustomCompose, transforms.Compose] = None,
    Processing: Processing = None,
    **kwargs,
):
    if not usePickle:
        assert callable(Processing), "Processing function is need!"
        dataset_ids: List[str] = os.listdir(preprocess_dir)
        dataset = Img5Dataset(dataset_ids, processing=Processing)
    else:
        assert isinstance(compose, (CustomCompose, transforms.Compose)), "compose must be CustomCompose or transforms.Compose"
        dataset = Pickle5Dataset(compose, preprocess_dir, filenames)

    train_len = int(len(dataset) * dataset_rate)
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    return train_set, val_set, train_loader, val_loader


def get_test_dataloader(
    test_preprocess_dir: str = str(DatasetInfo.data_dir),
    Processing=Processing,
    batch_size: int = 32,
    num_workers: int = 8,
):
    dataset_ids: List[str] = os.listdir(test_preprocess_dir)
    dataset = Img5Dataset(dataset_ids, processing=Processing)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return dataset, loader


def __data2pickle():
    dataset_ids: List[str] = os.listdir(str(DatasetInfo.data_dir))
    img5dataset = Img5Dataset(dataset_ids, processing=Processing2Tensor(), isTrain=True)

    data_len = len(img5dataset)

    start_id = 0  # int(data_len // 8 * 7)
    end_id = data_len  # int(data_len // 8 * 8)
    for i in range(start_id, end_id):
        path = f'{DatasetInfo.dataset_pickle_dir}/{i}.pickle'
        if not Path(path).exists():
            save_pickle(img5dataset[i], path)
        else:
            print(f"file exists: {path}")

        if i % 50 == 0:
            print(i)


def testData2pickle(test_dir: str):
    DatasetInfo.data_dir = Path(test_dir)
    dataset_ids: List[str] = sorted(os.listdir(test_dir))
    img5dataset = Img5Dataset(dataset_ids, processing=Processing2Tensor(), isTrain=False)

    data_len = len(img5dataset)

    start_id = 0  # int(data_len // 8 * 7)
    end_id = 183  # int(data_len // 8 * 8)
    for i in range(start_id, end_id):
        data = img5dataset[i]
        path = str(img5dataset.data_dir / f'{img5dataset.frame5_start}.pickle')
        save_pickle(data, path)

        if (i) % 50 == 0:
            print(i)


def separate_pickle5HitMiss():
    dataset = Pickle5Dataset(str(DatasetInfo.dataset_pickle_dir), None)
    loader = DataLoader(
        dataset,
        batch_size=30,
        num_workers=30,
        shuffle=False,
    )

    # hit_miss_table = np.zeros(len(dataset), dtype=np.uint8)

    import os
    from tqdm import tqdm
    from lib.FileTools.FileSearcher import check2create_dir

    hit_miss_dir = [str(DatasetInfo.dataset_pickle_dir / 'hit'), str(DatasetInfo.dataset_pickle_dir / 'miss')]
    check2create_dir(hit_miss_dir[0])
    check2create_dir(hit_miss_dir[1])

    dataset.filenames.sort(reverse=True)

    already_in = [
        *get_filenames(hit_miss_dir[0], '*.pickle', withDirPath=False),
        *get_filenames(hit_miss_dir[1], '*.pickle', withDirPath=False),
    ]
    for i in tqdm(range(len(dataset))):
        filename = dataset.filenames[i].split('/')[-1]

        if filename not in already_in:
            label: torch.Tensor = dataset[i][1]
            check = torch.argmax(label[:6]) // 5
            path = f'{hit_miss_dir[check]}/{filename}'

            os.symlink(f'../{filename}', f'{hit_miss_dir[check]}/{filename}')


if __name__ == '__main__':
    testData2pickle('Data/part1/private_data')

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)

#     sizeHW = (640, 640)

#     # for test stage
#     test_compose = CustomCompose(
#         [
#             transforms.GaussianBlur([3, 3]),
#             transforms.ConvertImageDtype(torch.float32),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
#         ]
#     )

#     argumentation_order_ls = [
#         RandomHorizontalFlip(p=0.5),
#         transforms.GaussianBlur([3, 3]),
#         transforms.RandomApply([transforms.ColorJitter(brightness=0.4, hue=0.2, contrast=0.5, saturation=0.2)], p=0.75),
#         transforms.RandomPosterize(6, p=0.15),
#         transforms.RandomEqualize(p=0.15),
#         transforms.RandomSolarize(128, p=0.1),
#         transforms.RandomInvert(p=0.05),
#         transforms.RandomApply(
#             [transforms.ElasticTransform(alpha=random.random() * 200.0, sigma=8.0 + random.random() * 7.0)], p=0.75
#         ),
#         RandomRotation(degrees=[-5, 5], p=0.75),
#         RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
#     ]
#     # for train & val stage
#     train_compose = CustomCompose(
#         [
#             *argumentation_order_ls,
#             transforms.ConvertImageDtype(torch.float32),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
#         ]
#     )

#     # for pre-view stage
#     view_compose = CustomCompose(argumentation_order_ls)

#     # train_set, val_set, train_loader, val_loader = get_dataloader(
#     #     preprocess_dir=str(DatasetInfo.data_dir),
#     #     dataset_rate=0.8,
#     #     batch_size=32,
#     #     num_workers=0,
#     #     Processing=Processing(view_compose),
#     # )

#     train_set, val_set, train_loader, val_loader = get_dataloader(
#         preprocess_dir=str(DatasetInfo.dataset_pickle_dir),
#         dataset_rate=0.8,
#         batch_size=32,
#         num_workers=0,
#         usePickle=True,
#         compose=view_compose,
#     )

#     w = 10
#     h = 10
#     columns = 5
#     rows = 3

#     i = 0
#     fig = plt.figure()

#     data: torch.Tensor
#     label: torch.Tensor  # [6]=0 -> hit | [6]=1 -> miss
#     for j, (data, label) in enumerate(val_loader):
#         batch_imgs = data.numpy().transpose(0, 1, 3, 4, 2)
#         for k, imgs in enumerate(batch_imgs):
#             for l, img in enumerate(imgs):
#                 fig.add_subplot(rows, columns, l + 5 * i + 1)
#                 plt.imshow(img)

#             i += 1
#             if i == 3:
#                 print(f"saving out/argumentation_view/{j}_{k}.png")
#                 plt.savefig(f'out/argumentation_view/{j}_{k}.png')

#                 i = 0
#                 fig = plt.figure()

#         # cv2.imwrite(f'tt{i}.jpg', cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_BGR2RGB))

#     # dataset_ids: List[str] = os.listdir(str(DatasetInfo.data_dir))
#     # dataset = Img5Dataset(dataset_ids, processing=Processing(compose))

#     # check_start = len(dataset) - (len(dataset) // 4)
#     # check_end = len(dataset)
#     # error_ids = []
#     # error_msgs = []
#     # for i in range(check_start, check_end):
#     #     try:
#     #         data, label = dataset[i]
#     #     except Exception as e:
#     #         error_msg = f'{i}, {e}'
#     #         print(str_format(error_msg, fore='r'))
#     #         error_ids.append(i)
#     #         error_msgs.append(error_msg)

#     #     if i % 100 == 0:
#     #         print(i)

#     # # dataset[check_start]

#     # # print(len(dataset))
#     # # print(str_format(f"error_ids: {error_ids}", fore='y'))
#     # # print(error_msgs)

#     # train_len = int(len(dataset) * 0.8)
#     # train_set, val_set = random_split(dataset, len(dataset) - train_len)

#     # dataset[1]
