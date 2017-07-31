import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pprint import pprint
from data import load_rows, make_output_matrix


def extract(data):
    classes = data.shape[2]
    result = []

    for c in range(0, classes):
        result.append([])
        while True:
            box = extract_pass(data, 0.9, c)
            if box is None:
                break
            else:
                result[c].append(box)
    return result


def extract_pass(matrix, treshold, c):
    size = matrix.shape[0]
    result = False

    sx = 0
    sy = 0
    ex = size
    ey = size

    has_sx = False
    has_ex = False
    has_sy = False
    has_ey = False

    has_row = False
    cVal = False
    pVal = False

    for y in range(max(sy - 1, 0), min(ey, size)):
        has_row = False
        pVal = False
        for x in range(max(sx - 1, 0), min(ex, size)):
            val = matrix[y][x][c]
            cVal = val >= treshold

            if cVal:
                matrix[y][x][c] = max(0.0, val - 1)
                has_row = True
                if not has_sx:
                    has_sx = True
                    sx = x
                else:
                    sx = min(sx, x)

                if has_ex and x > ex:
                    ex = x

            elif pVal:
                if has_ex:
                    ex = max(ex, x)
                else:
                    has_ex = True
                    ex = x
                break

            pVal = cVal
        if has_row:
            if not has_sy:
                has_sy = True
                sy = y
        elif has_sx:
            ey = y
            break

    if not has_sx:
        return None

    left_offset = 0
    right_offset = 0
    top_offset = 0
    bottom_offset = 0


    # TODO calc offsets (redo this in something that is not a 5 minute hackjob)
    if sx > 0: # Left Edge
        lsum = 0
        for y in range(sy, ey):
            lval = matrix[y][sx - 1][c]
            lval = lval - math.floor(lval)
            if lval > 0.25:
                matrix[y][sx - 1][c] -= lval
                lsum += lval
        left_offset = lsum / (ey - sy)

    if ex < size - 1: # Right Edge
        rsum = 0
        for y in range(sy, ey):
            rval = matrix[y][ex][c]
            rval = rval - math.floor(rval)
            if rval > 0.25:
                matrix[y][ex][c] -= rval
                rsum += rval
        right_offset = rsum / (ey - sy)

    if sy > 0: # Top Edge
        tsum = 0
        for x in range(sx, ex):
            tval = matrix[sy - 1][x][c]
            tval = tval - math.floor(tval)
            if tval > 0.25:
                matrix[sy - 1][x][c] -= tval
                tsum += tval
        top_offset = tsum / (ex - sx)

    if ey < size - 1: # Bottom Edge
        bsum = 0
        for x in range(sx, ex):
            bval = matrix[ey][x][c]
            bval = bval - math.floor(bval)
            if bval > 0.25:
                matrix[ey][x][c] -= bval
                bsum += bval
        bottom_offset = bsum / (ex - sx)

    sx -= left_offset
    ex += right_offset
    sy -= top_offset
    ey += bottom_offset

    return (sx / size, sy / size, (ex - sx) / size, (ey - sy) / size)

# Testing
X, y = load_rows(3, 1)
yvis = np.copy(y)
result = extract(y[0])
fig,ax = plt.subplots(1)
pprint(result)

ax.imshow(make_output_matrix(yvis[0], 0), cmap="gray")

for j in range(0, len(result[0])):
    bx, by, bw, bh = result[0][j]
    rect = patches.Rectangle((bx*11.0 - 0.5,by*11.0 - 0.5),bw*11.0,bh*11.0,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

plt.show()
