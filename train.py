import wandb
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import visualkeras
import os
from PIL import Image, ImageFilter
from numpy import asarray
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.init(
    project="flappy-bot",
    entity="tristan-retali",
    # (optional) set entity to specify your username or team name
    # entity="my_team",
    config={
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 10,
        "batch_size": 256,
    },
)
config = wandb.config

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
        img = Image.open(f).convert('L').resize((50, 50))
        numpydata = asarray(img)
        for i in range(JUMP_WEIGHT):
            xs.append(numpydata)
            ys.append(1)

for filename in os.listdir(f"{DATASET_DIR}/no_jump"):
    f = os.path.join(f"{DATASET_DIR}/no_jump", filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = Image.open(f).convert('L').resize((50, 50))
        numpydata = asarray(img)
        xs.append(numpydata)
        ys.append(0)

xs = np.array(xs)
ys = np.array(ys)

print(f"number of images: {xs.shape[0]}")
print(f"number of labels: {ys.shape[0]}")

# Model / data parameters
input_shape = (50, 50, 1)


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

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
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
        layers.Conv2D(64, kernel_size=(3, 3),
                      strides=(2, 2), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.01)),
        layers.Dense(2, activation="softmax"),
    ]
)
if TUNE:
    model = keras.models.load_model("./model")

model.summary()

visualkeras.layered_view(
    model, legend=True, to_file="./img/neural_network.png").show()

model.compile(loss=config.loss,
              optimizer=config.optimizer, metrics=[config.metric])

# Add WandbMetricsLogger to log metrics and WandbModelCheckpoint to log model checkpoints
wandb_callbacks = [
    WandbMetricsLogger(),
    # WandbModelCheckpoint(filepath="flappybot_model_{epoch:02d}"),
]
model.fit(x_train, y_train, epochs=config.epoch, batch_size=config.batch_size, validation_split=0.1,
          callbacks=wandb_callbacks)
wandb.finish()

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save("model")
