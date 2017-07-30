from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import math
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from config import ann_root, img_root


def build_generator():
    pass

def load_rows(offset, length):
    df = pd.read_pickle("data/cmap.p")
    X = np.zeros((length, 416, 416, 3), dtype=np.float32)
    y = np.zeros((length, 11, 11, 3), dtype=np.float32)

    for i in range(0, length):
        row = df.iloc[offset + i]
        X[i] = load_image(row["file"], 416, 416)
        y[i] = np.asarray(row["cmap"], dtype=np.float32)

    X = X / 255.0

    return (X, y)


def load_image(path, w, h):
    print(path)
    img = cv2.imread(img_root + path + ".jpg")
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h))
    return img


def make_output_matrix(data, c):
    img = np.zeros((11,11), dtype=np.float32)
    for y in range(0,11):
        for x in range(0,11):
            img[y][x] = data[y][x][c]
    return img

def vis_output(actual, data, c):
    max_a = data.max()
    data /= max_a
    actual /= max_a
    size = actual.shape[0]
    detectors = actual.shape[3]
    fig = plt.figure()
    n = 1
    print ("Showing %d maps with a AMAX of %d" % (size, max_a))

    for i in range(0, size):
        a = plt.subplot(size,2,n)
        a.imshow(make_output_matrix(actual[i], c), cmap="gray")
        n+=1

        a = plt.subplot(size,2,n)
        a.imshow(make_output_matrix(data[i], c), cmap="gray")
        n+=1
    plt.show()



#load_image("2007_000027", 416, 416)
#X,y = load_row(0)
#print(X)
#print(y.shape)
