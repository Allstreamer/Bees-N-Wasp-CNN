import os

import numpy as np
import tensorflow as tf
from matplotlib.pyplot import imshow, show

from Model import create_model

np.set_printoptions(precision=3)

IMG_SIZE = 128
loaded_data = np.load("data.npz")
x_train = loaded_data['X'] / 255
x_train = np.reshape(x_train,(-1,IMG_SIZE,IMG_SIZE,1))
y_train = loaded_data['Y']

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

checkpoint_savePath = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_savePath)
checkpoint_Callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_savePath,
                                                         save_weights_only=True,
                                                         verbose=1)

model = create_model()

try:
    model.load_weights(checkpoint_savePath)
except:
    print("No Weight Checkpoint")

model.fit(x_train, y_train,
        epochs=20,
        batch_size=16,
        shuffle=True,
        use_multiprocessing=True, workers=2,
        callbacks=[checkpoint_Callback])

print(model.predict(x_train)[1])
print(y_train[1])
imshow(x_train[1],cmap="gray")
show()