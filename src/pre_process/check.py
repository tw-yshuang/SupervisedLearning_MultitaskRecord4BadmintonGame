import os, subprocess
from pathlib import Path

import numpy as np
import cv2

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from submodules.UsefulTools.FileTools.FileOperator import get_filenames


main_dir = './Data/Paper_used/train'
id_str_ls = sorted(os.listdir(main_dir))[-150:]

target_dir = 'val'
for i, id_str in enumerate(id_str_ls):
    if i > 100:
        target_dir = 'test'
    subprocess.run(f'ln -s ../../part1/train/{id_str} ./Data/Paper_used/{target_dir}/{id_str}'.split(' '))
    subprocess.run(f'rm ./Data/Paper_used/train/{id_str}'.split(' '))
