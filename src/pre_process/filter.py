from pathlib import Path

import numpy as np
import cv2

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from submodules.UsefulTools.FileTools.FileOperator import get_filenames
from submodules.UsefulTools.FileTools.PickleOperator import save_pickle


filenames = sorted(get_filenames('./Data/part1/train', '*.mp4'))
noise = []

for filename in filenames:
    cap = cv2.VideoCapture(filename)

    frame: np.ndarray
    ret, frame = cap.read()

    bgr: np.ndarray = frame[0:20, 150:210].mean(axis=(0, 1))
    isWrong = False
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if (bgr > 150).all() == True:
        for i in range(20):
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == ord(' '):
                isWrong = True
                break

        if isWrong is False:
            noise.append(filename)
            print(filename)


save_pickle(noise, './noise.pickle')
