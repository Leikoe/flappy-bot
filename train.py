import wandb
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
from PIL import Image, ImageFilter
from numpy import asarray
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.init(
    project="flappy-bot",
    entity="leo-paille",
    # (optional) set entity to specify your username or team name
    # entity="my_team",
    config={
        "dropout": 0.50,
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 50,
        "batch_size": 256,
    },
)
config = wandb.config


TUNE = False
JUMP_WEIGHT = 5

# jump = 1, no_jump = 0
xs = []
ys = []

DATASET_DIR = "./Flappy-bird-python-dataset/dataset"
filenames = os.listdir(DATASET_DIR)
runs = {}
for filename in filenames:
    f = os.path.join(DATASET_DIR, filename)
    if os.path.isfile(f):
        run_uuid, frame_idx, jump = filename.replace(".png", "").split("_")
        frame_idx, jump = int(frame_idx), int(jump)
        img = Image.open(f).convert('L').resize((80, 80))

        # Scale images to the [0, 1] range
        numpydata = asarray(img).astype("float32") / 255
        if run_uuid not in runs:
            runs[run_uuid] = []
        runs[run_uuid].append((numpydata, frame_idx, jump))

for run_uuid in runs:
    runs[run_uuid].sort(key=lambda x: x[1])

    for i in range(0, len(runs[run_uuid]) - 4, 4):
        four_consecutive_frames = [runs[run_uuid][i+0][0], runs[run_uuid][i+1][0], runs[run_uuid][i+2][0], runs[run_uuid][i+3][0]]
        four_consecutive_frames = np.array(four_consecutive_frames, dtype="float32")
        four_consecutive_frames = np.einsum("chw->hwc", four_consecutive_frames)
        xs.append(four_consecutive_frames)
        ys.append(runs[run_uuid][3+i][2])

xs = np.array(xs)
ys = np.array(ys)

print(f"number of images: {xs.shape[0]}")
print(f"number of labels: {ys.shape[0]}")

# Model / data parameters
input_shape = (80, 80, 4, 1)

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

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# plt.imshow(xs[0], interpolation='nearest')
# plt.show()

plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(xs[0][:,:,i], cmap=plt.cm.binary)
    plt.xlabel(ys[0])
plt.show()

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling3D(pool_size=(2, 2, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(config.dropout),
        layers.Dense(64, activation="relu"),
        # layers.Dropout(config.dropout),
        # layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)
if TUNE:
    model = keras.models.load_model("./model")

model.summary()

model.compile(loss=config.loss,
              optimizer=config.optimizer, metrics=[config.metric])

# Add WandbMetricsLogger to log metrics and WandbModelCheckpoint to log model checkpoints
wandb_callbacks = [
    WandbMetricsLogger(),
    # WandbModelCheckpoint(filepath="flappybot_model_{epoch:02d}"),
]
model.fit(x_train, y_train, epochs=config.epoch, batch_size=config.batch_size, validation_split=0.1, callbacks=wandb_callbacks)
wandb.finish()

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save("model")
