from keras.callbacks import EarlyStopping
import numpy as np
from model import build_model
from data import load_rows, vis_output
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Parameters
epochs = 100
batch_size = 5
patience = 10

# Model
model = build_model()

# Data Generator
X, y = load_rows(0, 100)

# Early stop
early_stopping_monitor = EarlyStopping(patience=patience, monitor="acc", mode="auto")

# Fit Model
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping_monitor], validation_split=0.25)

# Predict and visualize
data = model.predict(X[-5:])
vis_output(y[-5:], data)
