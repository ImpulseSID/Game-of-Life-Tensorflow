import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, InputLayer, Layer
from tensorflow.keras.models import Sequential

size = 128
full_size = (1, size, size, 1)

env = np.random.randint(0, 2, full_size)


class TorusPaddingLayer(Layer):
    def __init__(self, **kwargs):
        super(TorusPaddingLayer, self).__init__(**kwargs)
        top_row = np.zeros((1, size))
        bottom_row = np.zeros((1, size))
        top_row[0, -1] = 1
        bottom_row[-1, 0] = 1

        self.pre = tf.convert_to_tensor(np.vstack((top_row, np.eye(size), bottom_row)), dtype=tf.float32)
        self.pre = tf.expand_dims(self.pre, 0)
        self.pre = tf.expand_dims(self.pre, -1)
        self.pre_T = tf.transpose(self.pre)

    def call(self, inputs):
        return tf.einsum("abcd,ecfg,hfij->abij", self.pre, inputs, self.pre_T)


def kernel(shape, dtype=None):
    kernel = np.ones(shape)
    kernel[1, 1] = 0
    return tf.convert_to_tensor(kernel, dtype=dtype)


model = Sequential([
    InputLayer(input_shape=full_size[1:]),
    TorusPaddingLayer(),
    Conv2D(1, 3, padding="valid", activation=None, use_bias=False, kernel_initializer=kernel)
])

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(xlim=(0, size), ylim=(0, size))
render = plt.imshow(env.squeeze(), interpolation="none", cmap="binary")
plt.axis("off")
plt.gca().invert_yaxis()


def update(frame):
    global env
    if not plt.fignum_exists(fig.number):
        exit()

    neighbours = model(env)
    env = np.where((env & np.isin(neighbours, (2, 3))) | ((env == 0) & (neighbours == 3)), 1, 0)

    if frame % 50 == 0:
        noise = np.random.choice([0, 1], size=full_size, p=[0.99, 0.01])
        env = np.bitwise_xor(env, noise)

    render.set_array(env.squeeze())
    plt.pause(0.01)


frame = 0
while True:
    update(frame)
    frame += 1
