import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pprint import pprint
from ..data import load_rows, make_output_matrix
from ..bbox import extract

X, y = load_rows(9, 1)
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
