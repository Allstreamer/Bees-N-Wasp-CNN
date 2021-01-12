from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_model():
    model = Sequential()

    model.add(Conv2D(32,(8,8)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(16, (4, 4)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(24,activation="relu"))
    model.add(Dense(3,activation="softmax"))

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    return model