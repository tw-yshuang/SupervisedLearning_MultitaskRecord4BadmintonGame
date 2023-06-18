import os, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from src.data_process import DatasetInfo
from submodules.UsefulTools.FileTools.FileOperator import get_filenames


def order_data(dataset_infos: List[DatasetInfo], len_dataset: int):
    data_order_ls = []
    data_id2startIdx_arr = np.zeros((len_dataset, 2), dtype=np.uint32)
    continue_idx = 0
    for i, dataset_info in enumerate(dataset_infos):
        data_id2startIdx_arr[i] = dataset_info.id, continue_idx
        pickle_paths = [*dataset_info.hit_pickle_paths, *dataset_info.miss_pickle_paths]
        start_idxs = [int(pickle_path.split('/')[-1].split('.')[0]) for pickle_path in pickle_paths]
        data_order_ls.extend(start_idxs)
        continue_idx += len(pickle_paths)

    data_order_arr = np.array(data_order_ls, dtype=np.int64)

    return data_id2startIdx_arr, data_order_arr


def get_static_handcraft_table(target_dir: str, data_dir: str, sub_dir: str, side: int) -> Tuple[List[np.ndarray], np.ndarray]:
    data_ids = sorted(os.listdir(target_dir))

    video_startIDs_ls = [
        sorted(map(lambda x: int(x[:-7]), get_filenames(f'{data_dir}/{data_id}/{sub_dir}', '*.pickle', withDirPath=False)))
        for data_id in data_ids
    ]

    # Data/Paper_used/test
    handcraft_table_ls = []
    hand_add = 1 / (side * 2 + 1)
    hand_sub = hand_add
    side = 2  # * because vector method's pickle file is frame5

    for video_startIDs in video_startIDs_ls:
        idxs = np.array(video_startIDs) + side
        handcraft_table = torch.zeros(idxs.max() + side * 2, dtype=torch.float32)

        handcraft_table[idxs] += hand_add
        for s in range(1, side + 1):
            handcraft_table[idxs + s] += hand_add
            handcraft_table[idxs - s] += hand_add

        for idx in idxs:
            handcraft_table[idx - side * 2 : idx + side * 2] -= hand_sub
            handcraft_table[idx - side * 1 : idx + side * 1] += hand_sub
        handcraft_table_ls.append(handcraft_table)

    correspond_table = np.array([*map(int, data_ids)], dtype=np.int16)

    return handcraft_table_ls, correspond_table


if __name__ == '__main__':
    target_dir = 'Data/Paper_used/test'
    source_dir = 'Data/AIdea_used/pre-process'
    sub_dir = 'ball_mask5_dir'

    handcraft_table_ls, correspond_table = get_static_handcraft_table(target_dir, source_dir, sub_dir, 1)
