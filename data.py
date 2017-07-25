from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import math
import pandas as pd

ann_root = "C:/Dev/Datasets/VOCdevkit/VOC2012/Annotations/"
img_root = "C:/Dev/Datasets/VOCdevkit/VOC2012/JPEGImages/"


def build_generator():
    df = pd.read_csv("data/cmap.csv")

    
