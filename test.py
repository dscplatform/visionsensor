import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import load_model
from pprint import pprint
from data import load_rows, make_output_matrix
from bbox import extract

# Parameters
version = 1
offset = 0

# Model
model = load_model(("export/mdl_v%d.h5")%(version))

# Load Data
X, y_true = load_rows(offset, 1)

# Predict
y_pred = model.predict(X)

# Extract Bounding boxes
yvis = np.copy(y_pred)
result = extract(y_pred[0])
pprint(result)

# Visualize
fig,ax = plt.subplots(1)
ax.imshow(make_output_matrix(yvis[0], 0), cmap="gray")

for j in range(0, len(result[0])):
    bx, by, bw, bh = result[0][j]
    rect = patches.Rectangle((bx*11.0 - 0.5,by*11.0 - 0.5),bw*11.0,bh*11.0,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

plt.show()
