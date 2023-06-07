import os, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from submodules.UsefulTools.FileTools.FileOperator import get_filenames
from src.data_process import CSVColumnNames

CNs = CSVColumnNames


if __name__ == '__main__':
    source_main_dir = 'Data/Paper_used'
    target_main_dir = 'Data/Paper_used/pre_process'
    image_main_dir = 'Data/part1/private_data'

    source_sub_dir_name = 'images'
    target_sub_dir_name = 'pickle'

    filenames = sorted(get_filenames('Data/Paper_used', '*.csv', withDirPath=False))

    for filename in filenames:
        df = pd.read_csv(f'{source_main_dir}/{filename}')
        for i in range(len(df)):
            series = df.iloc[i]

            frame_id = series.at[CNs.HitFrame]

            frame_strat = frame_id - 6
            frame_strat = frame_id + 6

            # TODO: concat Array

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
