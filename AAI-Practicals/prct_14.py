import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(128, activation='relu'),
        layers.Reshape((4, 4, 8)),
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2),
                               padding='same', activation='relu'),
        layers.Conv2DTranspose(1, (4, 4), strides=(7, 7),
                               padding='same', activation='sigmoid')
    ])
    return model

generator = build_generator()
noise = tf.random.normal([1, 100])
generated_image = generator(noise)

plt.imshow(generated_image[0, :, :, 0], cmap='gist_ncar')
plt.axis('off')
plt.savefig("generated_image.png", bbox_inches='tight')
#plt.show()

