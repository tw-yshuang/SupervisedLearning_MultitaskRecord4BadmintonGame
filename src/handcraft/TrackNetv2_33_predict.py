import numpy as np
import cv2

import keras.backend as K
from keras.models import *
from keras.layers import *

import tensorflow as tf

BATCH_SIZE = 1
HEIGHT = 288
WIDTH = 512
# HEIGHT=360
# WIDTH=640
sigma = 2.5
mag = 1

array_to_img = tf.keras.utils.array_to_img
img_to_array = tf.keras.utils.img_to_array


# Loss function
def custom_loss(y_true, y_pred):
    loss = (-1) * (
        K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1))
        + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
    )
    return K.mean(loss)


def prepare_model(weights_path: str):
    model = load_model(weights_path, custom_objects={'custom_loss': custom_loss})
    model.summary()
    print('Beginning predicting......')

    return model


class TrackNetV2_33:
    def __init__(self, filename: str) -> None:
        self.model = prepare_model(filename)

    def predict(self, img1: np.ndarray, img2: np.ndarray, img3: np.ndarray):
        ratio = img1.shape[0] / HEIGHT

        points = np.zeros((3, 2), dtype=np.int16)
        masks = np.zeros((3, *img1.shape[:-1]), dtype=np.uint8)

        unit = []
        # Adjust BGR format (cv2) to RGB format (PIL)
        x1 = img1[..., ::-1]
        x2 = img2[..., ::-1]
        x3 = img3[..., ::-1]
        # Convert np arrays to PIL images
        x1 = array_to_img(x1)
        x2 = array_to_img(x2)
        x3 = array_to_img(x3)
        # Resize the images
        x1 = x1.resize(size=(WIDTH, HEIGHT))
        x2 = x2.resize(size=(WIDTH, HEIGHT))
        x3 = x3.resize(size=(WIDTH, HEIGHT))
        # Convert images to np arrays and adjust to channels first
        x1 = np.moveaxis(img_to_array(x1), -1, 0)
        x2 = np.moveaxis(img_to_array(x2), -1, 0)
        x3 = np.moveaxis(img_to_array(x3), -1, 0)
        # Create data
        unit.append(x1[0])
        unit.append(x1[1])
        unit.append(x1[2])
        unit.append(x2[0])
        unit.append(x2[1])
        unit.append(x2[2])
        unit.append(x3[0])
        unit.append(x3[1])
        unit.append(x3[2])
        unit = np.asarray(unit)
        unit = unit.reshape((1, 9, HEIGHT, WIDTH))
        unit = unit.astype('float32')
        unit /= 255
        y_pred = self.model.predict(unit, batch_size=BATCH_SIZE)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype('float32')
        h_pred = y_pred[0] * 255
        h_pred = h_pred.astype('uint8')
        for i in range(3):
            if np.amax(h_pred[i]) <= 0:
                continue

            # h_pred
            (cnts, _) = cv2.findContours(h_pred[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for j in range(len(rects)):
                area = rects[j][2] * rects[j][3]
                if area > max_area:
                    max_area_idx = j
                    max_area = area
            target = rects[max_area_idx]
            (cx_pred, cy_pred) = (int(ratio * (target[0] + target[2] / 2)), int(ratio * (target[1] + target[3] / 2)))

            points[i] = (cx_pred, cy_pred)
            cv2.circle(masks[i], (cx_pred, cy_pred), 5, 255, -1)

        return masks, points
