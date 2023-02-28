from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
from PIL import Image, ImageFilter
from numpy import asarray

TUNE = False
JUMP_WEIGHT = 4

# jump = 1, no_jump = 0
xs = []
ys = []

DATASET_DIR = "./Flappy-bird-python-dataset/dataset"
for filename in os.listdir(f"{DATASET_DIR}/jump"):
    f = os.path.join(f"{DATASET_DIR}/jump", filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = Image.open(f).convert('L').filter(ImageFilter.FIND_EDGES).resize((25, 35))
        numpydata = asarray(img)
        for i in range(JUMP_WEIGHT):
            xs.append(numpydata)
            ys.append(1)

for filename in os.listdir(f"{DATASET_DIR}/no_jump"):
    f = os.path.join(f"{DATASET_DIR}/no_jump", filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = Image.open(f).convert('L').filter(ImageFilter.FIND_EDGES).resize((25, 35))
        numpydata = asarray(img)
        xs.append(numpydata)
        ys.append(0)

xs = np.array(xs)
ys = np.array(ys)

print(f"number of images: {xs.shape[0]}")
print(f"number of labels: {ys.shape[0]}")

# Model / data parameters
input_shape = (35, 25, 1)

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

# plt.imshow(xs[0], interpolation='nearest')
# plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(xs[i], cmap=plt.cm.binary)
    plt.xlabel(ys[i])
plt.show()

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),
    ]
)
if TUNE:
    model = keras.models.load_model("./model")

model.summary()

epochs = 100
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_split=0.1)


score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save("model")
