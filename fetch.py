from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import xml.etree.ElementTree as ET
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from config import ann_root, img_root

def make_classmap(classes):
    return dict((classes[i],i) for i in range(0,len(classes)))

def make_dataset(classes, detectors, width, height, *args):
    classmap = make_classmap(classes)
    max_entries = args[0] if len(args) > 0 else -1
    y = []
    for f in os.listdir(ann_root):
        xml = ET.parse("%s%s" % (ann_root, f))
        data = {"file": f, "width": 0, "height": 0, "map": np.zeros((detectors, detectors, len(classes)), dtype=np.float32)}
        objects = []

        for e in xml.iter():
            if "width" in e.tag:
                data["width"] = int(e.text)
            elif "height" in e.tag:
                data["height"] = int(e.text)
            elif "object" in e.tag:
                obj = {}
                for attr in list(e):
                    if "name" in attr.tag and attr.text in classmap:
                        obj["class"] = classmap[attr.text]
                        objects.append(obj)
                    if "bndbox" in attr.tag:
                        for dim in list(attr):
                            if "xmin" in dim.tag:
                                obj["xmin"] = float(dim.text)
                            if "ymin" in dim.tag:
                                obj["ymin"] = float(dim.text)
                            if "xmax" in dim.tag:
                                obj["xmax"] = float(dim.text)
                            if "ymax" in dim.tag:
                                obj["ymax"] = float(dim.text)
        if len(objects) != 0:
            y.append(make_dataset_entry(data, objects, detectors))
        if max_entries != -1 and len(y) >= max_entries:
            return y
    return y

def make_dataset_entry(entry, objects, detectors):
    cmap = entry["map"]
    w = float(entry["width"])
    h = float(entry["height"])
    for obj in objects:
        c = obj["class"]
        sx = detectors * obj["xmin"] / w
        sy = detectors * obj["ymin"] / h
        ex = detectors * obj["xmax"] / w
        ey = detectors * obj["ymax"] / h

        # Inner box
        in_sx = math.ceil(sx)
        in_sy = math.ceil(sy)
        in_ex = math.floor(ex)
        in_ey = math.floor(ey)

        # Offset
        off_sx = max(0, in_sx - sx)
        off_sy = max(0, in_sy - sy)
        off_ex = max(0, ex - in_ex)
        off_ey = max(0, ey - in_ey)

        # Add to map
        for y in range(in_sy, in_ey):
            for x in range(in_sx, in_ex):
                cmap[y][x][c] += 1

        # Add offset to map
        for y in range(in_sy, in_ey):
            if off_sx > 0.25:
                cmap[y][in_sx - 1][c] += off_sx
            if off_ex > 0.25:
                cmap[y][in_ex][c] += off_ex

        for x in range(in_sx, in_ex):
            if off_sy > 0.25:
                cmap[in_sy - 1][x][c] += off_sy
            if off_ey > 0.25:
                cmap[in_ey][x][c] += off_ey

    return (entry["file"][:-4], cmap.tolist())


cmap = make_dataset(["person","bird","car"], 11, 416, 416, 100)
df = pd.DataFrame(cmap, columns=["file", "cmap"])
#df.to_csv("data/cmap.csv", index_label="id")
df.to_pickle("data/cmap.p")
