import math
import numpy as np


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
    sx = 0
    sy = 0
    ex = size
    ey = size
    has_sx = False
    has_ex = False
    has_sy = False
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

    left_offset = extract_offset(matrix[sy:ey,sx-1,c]) if sx > 0 else 0
    right_offset = extract_offset(matrix[sy:ey,ex,c]) if ex < size - 1 else 0
    top_offset = extract_offset(matrix[sy-1,sx:ex,c]) if sy > 0 else 0
    bottom_offset = extract_offset(matrix[ey,sx:ex,c]) if ey < size - 1 else 0

    sx -= left_offset
    ex += right_offset
    sy -= top_offset
    ey += bottom_offset

    return (
        sx / size,
        sy / size,
        (ex - sx) / size,
        (ey - sy) / size
    )


def extract_offset(column):
    size = len(column)
    avg = np.average(column)
    avg = avg - math.floor(avg)
    if avg > 0.25:
        column -= avg
        return avg
    return 0.0
