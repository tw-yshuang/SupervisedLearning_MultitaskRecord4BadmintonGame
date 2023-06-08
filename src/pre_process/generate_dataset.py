import os, subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import cv2
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from submodules.UsefulTools.FileTools.FileOperator import get_filenames, check2create_dir
from submodules.UsefulTools.FileTools.PickleOperator import save_pickle
from src.data_process import CSVColumnNames

CNs = CSVColumnNames


if __name__ == '__main__':
    source_main_dir = Path('Data/Paper_used')
    target_main_dir = Path('Data/Paper_used/pre_process/13seq')
    image_main_dir = Path('Data/AIdea_used/pre-process')

    source_sub_dir_name = 'image'

    filenames: List[str] = sorted(get_filenames(str(source_main_dir), '*.csv', withDirPath=False))

    for filename in filenames:
        data_type, data_id = filename.split('/')[:-1]
        target_dir = '/'.join([data_type, data_id])

        img_dir = image_main_dir / data_id / source_sub_dir_name
        target_dir = target_main_dir / target_dir
        if check2create_dir(str(target_dir)):
            continue

        df = pd.read_csv(str(source_main_dir / filename))
        max_frame_id = max(map(int, [path.replace('.jpg', '') for path in get_filenames(str(img_dir), '*.jpg', withDirPath=False)]))

        # create a cube that contains 13 sequence frames, 7th is the HitFrame
        # one pickle is contain: [cube, process_label] the format is all tensor
        for i in range(len(df)):
            series = df.iloc[i]

            frame_id = series.at[CNs.HitFrame]

            frame_start = frame_id - 6
            frame_end = frame_id + 6
            frame_seq_nums = np.zeros(13, dtype=np.int32)
            frame_seq_nums[:] = [*range(frame_start, frame_end + 1)]
            frame_seq_nums[frame_seq_nums < 1] = 1
            frame_seq_nums[frame_seq_nums > max_frame_id] = max_frame_id

            try:
                hit_frame = series.at[CSVColumnNames.HitFrame]

                process_label = torch.zeros(23, dtype=torch.float32)
                process_label[0 if series.at[CSVColumnNames.Hitter] == 'A' else 1] = 1.0  # Hitter: [0,1] one-hot
                process_label[1 + series.at[CSVColumnNames.RoundHead]] = 1.0  # RoundHead: [2,3] one-hot
                process_label[3 + series.at[CSVColumnNames.Backhand]] = 1.0  # Backhand: [4,5] one-hot
                process_label[5 + series.at[CSVColumnNames.BallHeight]] = 1.0  # BallHeight: [6,7] one-hot
                process_label[7 + series.at[CSVColumnNames.BallType]] = 1.0  # BallType: [8:16] one-hot
                process_label[17:23] = torch.from_numpy(
                    series.loc[
                        [
                            CSVColumnNames.LandingX,  # LandingX: 17
                            CSVColumnNames.LandingY,  # LandingY: 18
                            CSVColumnNames.HitterLocationX,  # HitterLocationX: 19
                            CSVColumnNames.HitterLocationY,  # HitterLocationY: 20
                            CSVColumnNames.DefenderLocationX,  # DefenderLocationX: 21
                            CSVColumnNames.DefenderLocationY,  # DefenderLocationY: 22
                        ],
                    ].to_numpy(dtype=np.float32)
                )
            except KeyError:
                os.rmdir(str(target_dir))
                break

            imgs = torch.tensor(
                np.stack([cv2.imread(str(img_dir / f'{idx}.jpg'), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for idx in frame_seq_nums])
            )

            save_pickle((imgs, process_label), str(target_dir / f'{frame_start}.pickle'))
