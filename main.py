import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.initializers import random_normal
import numpy as np


# print(np.random.random(100))
# print(keras.layers.Dense(1))

np.random.seed(42)

NUM_DATASETS = 1
DATASET_SIZE = 100000
NUM_FEATURES = 1
# input = np.array([[1.0]])
x = np.random.normal(0, 1, (NUM_DATASETS, DATASET_SIZE, NUM_FEATURES))
print(x.shape)
# print(input)
network = keras.Sequential(
    [
        Dense(
            1,
            kernel_initializer=random_normal(1, 1),
            bias_initializer=random_normal(1, 1),
        )
        for _ in range(1)
    ]
    + [
        Dense(
            1,
            kernel_initializer=random_normal(1, 1),
            bias_initializer=random_normal(1, 1),
        )
    ]
)
y = network.predict(x)
print(y.shape)

# plt.hist(y.flatten())
# print(np.std(y.flatten()))
# plt.show()
# plt.hist(x.flatten())
# print(np.std(x.flatten()))
# plt.show()

datasets = np.c_[(x, y)]
print(datasets.shape)
print(datasets[0, 0])


model = keras.Sequential([Dense(16, activation="relu") for _ in range(1)] + [Dense(1)])
model.compile(keras.optimizers.Adam(), loss="mse")
x, y = datasets[0, :, :NUM_FEATURES], datasets[0, :, NUM_FEATURES:]
print(x.shape, y.shape)
model.fit(x, y, validation_split=0.1, epochs=10, shuffle=True)
# model.fit(x[:500], y[:500], epochs=1000, validation_data=(x[500:], y[500:]))
print(model.evaluate(x, y))
