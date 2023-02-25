import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
from PIL import Image
from numpy import asarray

# jump = 1, no_jump = 0
xs = []
ys = []

DATASET_DIR = "dataset"
for filename in os.listdir(f"{DATASET_DIR}/jump"):
    f = os.path.join(f"{DATASET_DIR}/jump", filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = Image.open(f).convert('L')
        numpydata = asarray(img)
        xs.append(numpydata)
        ys.append(1)

for filename in os.listdir(f"{DATASET_DIR}/no_jump"):
    f = os.path.join(f"{DATASET_DIR}/no_jump", filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = Image.open(f).convert('L')
        numpydata = asarray(img)
        xs.append(numpydata)
        ys.append(0)

xs = np.array(xs)
ys = np.array(ys)

print(f"number of images: {xs.shape[0]}")
print(f"number of labels: {ys.shape[0]}")

from matplotlib import pyplot as plt
plt.imshow(xs[0], interpolation='nearest')
plt.show()

# Model / data parameters
input_shape = (600, 400, 1)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# shuffle the data
xs, ys = unison_shuffled_copies(xs, ys)

# Load the data and split it between train and test sets
TRAIN_SPLIT = 0.9
TRAIN_IDX = int(xs.shape[0] * TRAIN_SPLIT)
x_train, x_test = xs[:TRAIN_IDX], xs[TRAIN_IDX:]
y_train, y_test = ys[:TRAIN_IDX], ys[TRAIN_IDX:]
print(x_train.shape)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (600, 400, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

epochs = 15
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=epochs, validation_split=0.1)


score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
