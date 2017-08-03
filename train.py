from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from model import build_model
from data import load_rows, vis_output
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Parameters
version = 1
epochs = 200
batch_size = 1
patience = 25

# Model
model = build_model()

# Data Generator
X, y = load_rows(0, 50)

# Early stop
early_stopping_monitor = EarlyStopping(patience=patience, monitor="acc", mode="auto")

# Checkpoint
checkpoint_path = "export/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckPoint(checkpoint_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max")

# Fit Model
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping_monitor, checkpoint], validation_split=0.25)

# Export Model
model.save(("export/mdl_v%d.h5")%(version))
print("Model successfully exported!")
