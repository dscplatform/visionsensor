from model import build_model
from data import build_generator

# Parameters
epochs = 10
batch_size = 10
patience = 3

# Model
model = build_model()

# Data Generator
Xg, y = build_generator()
