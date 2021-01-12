import os

import cv2
import numpy as np

DATADIR = "./data"
CATEGORIES = ["bee","wasp","other"]
IMG_SIZE = 128

data_X = []
data_Y = []

def Create_Training_Data():
    for i,category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR,category)
        class_num = i
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            data_X.append(img_array)
            y = np.zeros(3)
            y[class_num] = 1
            data_Y.append(y)


Create_Training_Data()

np.savez("data.npz", X=data_X, Y=data_Y)