import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import keras
import numpy as np


# print(np.random.random(100))
# print(keras.layers.Dense(1))

np.random.seed(42)

# input = np.array([[1.0]])
x = np.random.normal(0, 1, (1000, 1000, 4))
print(x.shape)
# print(input)
network = keras.Sequential([keras.layers.Dense(16) for _ in range(4)] + [keras.layers.Dense(1)])
y = network.predict(x)
print(y.shape)

datasets = np.c_[(x, y)]
print(datasets.shape)
print(datasets[0][0])


# model = keras.Sequential([keras.layers.Dense(1)])
# model.compile(keras.optimizers.Adam(), loss="mse")
# model.fit(x, y, epochs=100)
