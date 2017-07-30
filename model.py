from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, Dropout, Activation, Input, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Nadam
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
from keras import backend as K

def build_model():
    model = Sequential()

    model.add(Conv2D(8, (3,3), strides=(1,1), padding="same", use_bias=False, input_shape=(416,416,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))

    for i in range(0,4):
      model.add(Conv2D(32*(2**i), (3,3), use_bias=False, strides=(1,1)))
      model.add(BatchNormalization())
      model.add(LeakyReLU(alpha=0.1))
      model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256, (3,3), padding="same", use_bias=False, strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    for i in range(0,2):
      model.add(Conv2D(512, (3,3), padding="same", use_bias=False, strides=(1,1)))
      model.add(BatchNormalization())
      model.add(Activation("relu"))

    model.add(Conv2D(8, (3,3), padding="same", use_bias=False, strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(11*11*3, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(11*11*3, activation="relu"))
    model.add(Reshape((11,11,3)))

    # TODO custom loss function(?)
    model.summary()
    model.compile(loss="mse", optimizer=Adam(), metrics=["accuracy"])
    return model


#model = build_model()
#plot_model(model, to_file="data/model.png")

def loss_func(y_true, y_pred):
    pass
