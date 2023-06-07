from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import cv2
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))

from src.handcraft.TrackNetv2_33_predict import TrackNetV2_33
from submodules.UsefulTools.FileTools.FileOperator import check2create_dir, get_filenames
from submodules.UsefulTools.FileTools.PickleOperator import save_pickle
from submodules.UsefulTools.FileTools.WordOperator import str_format


class TrackDebug:
    dir = Path('Data/part1/private_data')
    image_dir = Path('image')
    predict_dir = Path('predict')
    mask_dir = Path('mask')
    predict_merge_dir = Path('predict_merge')
    ball_mask5_dir = Path('ball_mask5_dir')
    HEIGHT = 288
    WIDTH = 512

    def __init__(self, data_id) -> None:
        self.dir = TrackDebug.dir / data_id
        self.image_dir = self.dir / TrackDebug.image_dir
        self.image_dir = self.dir / TrackDebug.image_dir
        self.mask_dir = self.dir / TrackDebug.mask_dir
        self.predict_dir = self.dir / TrackDebug.predict_dir
        self.predict_merge_dir = self.dir / TrackDebug.predict_merge_dir
        self.ball_mask5_dir = self.dir / TrackDebug.ball_mask5_dir

        check2create_dir(str(self.dir))
        check2create_dir(str(self.image_dir))
        check2create_dir(str(self.predict_dir))
        check2create_dir(str(self.mask_dir))
        check2create_dir(str(self.predict_merge_dir))
        check2create_dir(str(self.ball_mask5_dir))


class TrackPoint:
    def __init__(self, img_size: Tuple[int], num_frame: int, model: TrackNetV2_33, debugger: Union[TrackDebug, None] = None) -> None:
        '''
        This is the initialization function for a class that stores image size, number of frames, a model,
        and debugging information.

        Args:
            `img_size` (Tuple[int]): A tuple of integers representing the size of the input images to the model.

            `num_frame` (int): The number of frames to be processed by the algorithm.

            `model` (TrackNetV2_33): The model parameter is an instance of the TrackNetV2_33 class, which is a deep learning model used for object tracking in videos.

            `debugger` (Union[TrackDebug, None]): The `debugger` parameter is an optional argument that can be passed to the constructor. It is an instance of the `TrackDebug` class or `None`. If it is not `None`, it is used for debugging purposes during the tracking process.
        '''
        self.img_size = img_size
        self.num_frame = num_frame
        self.model = model
        self.debugger = debugger

        self.img3_arr: np.ndarray = np.zeros((3, *self.img_size, 3), dtype=np.uint8)  # 3 means the model input 3 images for once
        self.img3order_arr: np.ndarray = np.zeros_like(self.img3_arr, dtype=np.uint8)  # the according order for the self.img2_arr
        self.idx0, self.idx1, self.idx2 = 0, 0, 0

        self.currentFrame = 0

        self.masks: np.ndarray
        self.points: np.ndarray
        self.total_masks3 = np.zeros((self.num_frame + 2, 3, *self.img_size), dtype=np.uint8)  # 3 means 1 frame predict 3 times
        self.total_points3 = np.zeros((self.num_frame, 3, 2), dtype=np.uint16)  # 3 means 1 frame predict 3 times

    def update_frame(self, frame: np.ndarray, isDebug=False):
        '''
        This function updates a frame by shifting and updating an array of three frames.

        Args:
            `frame` (np.ndarray): a numpy array representing a single frame of an image or video.
        '''
        self.idx2 = self.idx1
        self.idx1 = self.idx0
        self.idx0 = self.currentFrame % 3

        self.currentFrame += 1

        self.img3_arr[self.idx0] = frame
        self.img3order_arr[:] = [self.img3_arr[self.idx2], self.img3_arr[self.idx1], self.img3_arr[self.idx0]]

        if isDebug:
            cv2.imwrite(str(self.debugger.image_dir / f'{self.currentFrame}.jpg'), frame)

    def predict(self, isDebug=False):
        '''
        This function predicts masks and points for 3 sequence images and displays the result in a debug window.

        Args:
            `isDebug`: isDebug is a boolean parameter that is used to determine whether to display debug information or not. If it is set to True, the function will display debug information, otherwise it will not. Defaults to False

        Returns:
            nothing (i.e., `None`). It is only performing some operations and displaying images if `isDebug` is set to `True`.
        '''
        if self.currentFrame < 3:
            return

        self.masks, self.points = self.model.predict(*self.img3order_arr)
        # print(self.currentFrame, self.points)
        orderIDs = (np.arange(self.currentFrame - 3, self.currentFrame), [2, 1, 0])
        self.total_masks3[orderIDs] = self.masks
        self.total_points3[orderIDs] = self.points
        # print(self.total_points3[orderIDs])
        # print(self.total_points3[orderIDs].shape)

        if isDebug is True:
            for i in range(3):
                img_cp = self.img3order_arr[i].copy()
                if self.masks is not None:
                    img_cp[self.masks[i] > 100] = (0, 255, 0)

                # cv2.imshow(f'img{i}', cv2.resize(img_cp, (self.debugger.WIDTH, self.debugger.HEIGHT)))
                cv2.imwrite(str(self.debugger.predict_dir / f'{self.currentFrame-2+i}_{2-i}.jpg'), img_cp)

    def get_rangeInfo(self, start_frame: int = 0, end_frame: int = -1):
        '''
        This function returns range information based on a given start and end frame.

        Args:
            `start_frame` (int): The starting frame number from which the range information needs to be calculated. If not provided, it defaults to 0. Defaults to 0

            `end_frame` (int): The end_frame parameter is an optional integer that specifies the last frame to include in the range of frames for which to retrieve information. If it is not provided or is set to -1, the function will use the currentFrame attribute of the object as the end frame.

        Returns:
            The function `get_rangeInfo` returns a tuple containing two elements: `points_arr` and `hit_frames`. `points_arr` is an array of estimated points, and `hit_frames` is an array of frames where the points were estimated.
        '''
        start_frame = 0 if start_frame == 0 else start_frame
        end_frame = self.currentFrame if end_frame == -1 else end_frame

        points3_arr = self.total_points3[start_frame:end_frame]

        points_arr, hit_frames = self.estimate_rangeInfo(points3_arr)
        hit_frames += start_frame

        return points_arr, hit_frames

    @staticmethod
    # @jit(nopython=True)
    def estimate_rangeInfo(points3_arr: np.ndarray):
        '''
        The function estimates the badminton coordinates and identifies frames where a hit occurred based on
        a given array of each frame predict 3 points from the model.

        Args:
            `points3_arr` (np.ndarray): The input parameter is a numpy array of shape (n, 3, 2), where n is the number of frames in a badminton game. Each element in the array represents the (x, y) coordinates of three points on the badminton court in a particular frame.

        Returns:
            The function `estimate_rangeInfo` returns a tuple containing two numpy arrays:
        1. `points_arr`: an array of shape `(n, 2)` containing the estimated coordinates of the badminton
        points.
        2. `hit_frames`: an array of shape `(m,)` containing the frame numbers where a hit is detected based
        on the change in distance between consecutive badminton points.
        '''

        points_arr = np.zeros(points3_arr.shape[0::2], dtype=np.float32)  # the estimate of the badminton coordinates

        for i, point3 in enumerate(points3_arr):
            points = point3[np.nonzero(np.sum(point3, axis=1))]
            points_arr[i] = np.mean(points, axis=0)

        nan_idxs = np.isnan(points_arr[:, 0])
        no_nan_idxs = np.logical_not(nan_idxs)

        for i in range(2):
            points_arr[nan_idxs, i] = np.interp(nan_idxs.nonzero()[0], no_nan_idxs.nonzero()[0], points_arr[no_nan_idxs, i])

        # bb = aa[no_nan_idxs] - points_arr[no_nan_idxs]

        vectors_arr = points_arr[:-1] - points_arr[1:]
        vectors_dist_arr = np.linalg.norm(vectors_arr[:-1] - vectors_arr[1:], axis=1)

        hit_frames = np.where(vectors_dist_arr[1:] - vectors_dist_arr[:-1] > 10)[0] + 2

        return np.uint16(points_arr), hit_frames

    def get_hitRangeMasks5(self, isDebug=False):
        '''
        This function generates 5 masks for each estimated frame of a badminton hit, with the center of each
        mask being the hit frame and the masks being a combination of circles and previously generated
        masks.

        Returns:
            The function `get_hitRangeMasks5` returns a tuple containing two elements:
        1. A list of numpy arrays, where each numpy array is a mask of shape (5, *self.img_size) representing the badminton court area for each hit frame. The list contains as many masks as there are hit frames.
        2. An array of each masks5 corresponding start frame.
        '''

        points_arr, hit_frames = self.get_rangeInfo()
        masks5_ls = []

        for hit_frame in hit_frames:  # hit_frames are the estimated frame numbers of badminton that are hit
            masks5 = np.zeros((5, *self.img_size), dtype=np.uint16)  # create 5 mask for corresponding frame, center is the hit_frame
            for i in range(-2, 3):
                if points_arr.shape[0] <= (hit_frame + i):
                    continue
                cv2.circle(masks5[i + 2], points_arr[hit_frame + i], 5, 255, -1)

            # cv2.imshow(f'hit_frame: {hit_frame}', np.uint8(masks5[2]))

            masks5 += np.sum(self.total_masks3[hit_frame - 2 : hit_frame + 3], axis=1)

            masks5_ls.append(np.uint8(masks5))

            # if isDebug:
            #     cv2.imshow(f'hit_frame: {hit_frame} merge', np.uint8(masks5[2]))

        if isDebug:
            for i, point_arr in enumerate(points_arr, start=1):
                mask = np.zeros(self.img_size, dtype=np.uint16)
                cv2.circle(mask, point_arr, 5, 255, -1)
                mask += np.sum(self.total_masks3[i - 1], axis=0)
                cv2.imwrite(str(self.debugger.mask_dir / f'{i}.jpg'), np.uint8(mask))

                img = cv2.imread(str(self.debugger.image_dir / f'{i}.jpg'))
                img[mask > 0] = (255, 255, 255)
                cv2.imwrite(str(self.debugger.predict_merge_dir / f'{i}.jpg'), np.uint8(img))

        return masks5_ls, hit_frames - 2  # the list of the masks5, and each masks5 corresponding start frame


if __name__ == '__main__':
    DEBUG_LS = [
        'update_frame',
        'predict',
        'get_hitRangeMasks5',
    ]

    data_dir = 'Data/part1/private_data'

    with tf.device('/gpu:0'):
        tNet33 = TrackNetV2_33('src/TrackNetv2/3_in_3_out/model906_30')

        # for multiple dir
        filenames: List[str] = get_filenames(data_dir, '*.mp4', withDirPath=False)
        filenames.sort(reverse=True)

        # import os
        # already_filenames = sorted(os.listdir('Data/result'), reverse=True)
        # [filenames.pop(int(f) - 1) for f in already_filenames]

        # filenames = filenames[len(filenames) // 2 :]

        # for one dir test
        # DEBUG_LS = []
        # filenames = ['00041/00041.mp4']

        # filenames = filenames[149:230]
        # filenames = filenames[350:]
        # filenames = filenames[330:550] 317, 159
        # filenames = filenames[550:]
        # filenames = sorted(filenames[:239], reverse=True)
        rest_data = 0
        for filename in filenames:
            data_id = filename.split('/')[0]
            debugger = TrackDebug(data_id)

            # checker
            check_ls = get_filenames(str(debugger.predict_merge_dir), '*.jpg')
            if len(check_ls) != 0:
                continue
            rest_data += 1

            print(str_format(filename, fore='y'))

            cap = cv2.VideoCapture(f'{data_dir}/{filename}')
            FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            HIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            tp = TrackPoint((HIGHT, WIDTH), FRAME, model=tNet33, debugger=debugger)

            for _ in range(FRAME):
                ret, frame = cap.read()
                if ret is False:
                    tp.currentFrame += 1
                    continue
                tp.update_frame(frame, isDebug='update_frame' in DEBUG_LS)
                tp.predict(isDebug='predict' in DEBUG_LS)

            masks5_ls, mask5startFrames = tp.get_hitRangeMasks5(isDebug='get_hitRangeMasks5' in DEBUG_LS)

            for mask5, mask5startFrame in zip(masks5_ls, mask5startFrames):
                save_pickle(mask5, f'{debugger.ball_mask5_dir}/{mask5startFrame}.pickle')

            cap.release()

    print(str_format(f"rest: {rest_data}", fore='y'))
