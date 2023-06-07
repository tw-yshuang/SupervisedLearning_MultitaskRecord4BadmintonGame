import os, subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from lib.FileTools.FileSearcher import get_filenames
from lib.FileTools.PickleOperator import load_pickle

noises = sorted(load_pickle('filter/noise.pickle'), reverse=True)

filenames = sorted(os.listdir('./Data/part1/train'))

nosie_dir = '/'.join(noises.pop().split('/')[:-1])
noise_id_str = nosie_dir.split('/')[-1]
for id_str in filenames:
    if noise_id_str == id_str:
        if len(noises) != 0:
            nosie_dir = noises.pop()
            noise_id_str = nosie_dir.split('/')[-2]
        continue

    subprocess.run(f'ln -s ../../part1/train/{id_str} ./Data/Paper_used/train/{id_str}'.split(' '))
