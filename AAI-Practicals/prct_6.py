import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)

BUFFER_SIZE = x_train.shape[0]
BATCH_SIZE = 128
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_generator():
    return tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(28 * 28, activation="sigmoid"),
        layers.Reshape((28, 28, 1))
    ])

def build_discriminator():
    return tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

discriminator.trainable = False

gan_input = layers.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer="adam", loss="binary_crossentropy")

EPOCHS = 3

for epoch in range(EPOCHS):
    for real_images in dataset:
        batch_size = real_images.shape[0]

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise, verbose=0)

        discriminator.train_on_batch(real_images, real_labels)
        discriminator.train_on_batch(fake_images, fake_labels)

        noise = np.random.normal(0, 1, (batch_size, 100))
        gan.train_on_batch(noise, real_labels)

    print(f"Epoch {epoch+1}/{EPOCHS} completed")

noise = np.random.normal(0, 1, (1, 100))
generated_image = generator.predict(noise, verbose=0)

plt.imshow(generated_image[0, :, :, 0], cmap="gray")
plt.axis("off")
plt.savefig("gan_generated_image.png", bbox_inches="tight")
#plt.show()

