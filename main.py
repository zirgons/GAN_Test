import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Keras modules from tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Conv2DTranspose, Reshape, BatchNormalization, Dropout, Input, ReLU, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.losses import BinaryCrossentropy
from PIL import Image
from tensorflow.keras.initializers import RandomNormal


import warnings
warnings.filterwarnings('ignore')
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

img_width, img_height = 256, 256
batchsize = 32

train = image_dataset_from_directory(
    directory='C:/Users/karli/Documents/GAN/archive/',
    batch_size=batchsize,
    image_size=(img_width, img_height)
)

data_iterator = train.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(10,10))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


DIR = 'C:/Users/karli/Documents/GAN/archive/' #path

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(
        DIR,
        target_size = (64, 64),
        batch_size = batchsize,
        class_mode = None)

KI = RandomNormal(mean=0.0, stddev=0.02)
input_dim = 512  

def Generator_Model():
    model = Sequential()

    # Random noise input
    model.add(Dense(128 * 16 * 16, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16, 16, 128)))

    # Upsampling layers
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=KI))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=KI))
    model.add(LeakyReLU(alpha=0.2))

    # Output layer for generated image
    model.add(Conv2D(3, (4, 4), padding='same', activation='tanh'))  # Output shape: (64, 64, 3)

    return model

generator = Generator_Model()
generator.summary()
tf.keras.utils.plot_model(generator, show_shapes=True)

from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend as K

class SpectralNormalization(Layer):
    def __init__(self, layer, **kwargs):
        self.layer = layer
        super(SpectralNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.u = self.add_weight(
            shape=(1, self.layer.kernel.shape[-1]),
            initializer=RandomNormal(mean=0.0, stddev=0.02),
            trainable=False,
            name='sn_u',
        )

    def call(self, inputs, training=None):
        w = self.layer.kernel
        w_shape = w.shape.as_list()
        w_reshaped = tf.reshape(w, [-1, w_shape[-1]])

        u_hat = self.u
        for _ in range(1):  # Iteratively calculate u and v
            v_hat = K.l2_normalize(K.dot(u_hat, K.transpose(w_reshaped)))
            u_hat = K.l2_normalize(K.dot(v_hat, w_reshaped))

        sigma = K.dot(K.dot(v_hat, w_reshaped), K.transpose(u_hat))
        w_norm = w / sigma

        self.u.assign(u_hat)
        self.layer.kernel = tf.reshape(w_norm, w_shape)
        return self.layer(inputs, training=training)

# Updated Discriminator Model
def Discriminator_Model(input_shape=(64, 64, 3)):
    model = Sequential()

    model.add(SpectralNormalization(Conv2D(64, (3, 3), input_shape=input_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpectralNormalization(Conv2D(128, (3, 3))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpectralNormalization(Conv2D(256, (3, 3))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = Discriminator_Model()
discriminator.summary()
tf.keras.utils.plot_model(discriminator, show_shapes=True)

from tensorflow.keras import Model
import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

# Add gradient penalty
def gradient_penalty(discriminator, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated_images)
        pred = discriminator(interpolated_images, training=True)

    grads = gp_tape.gradient(pred, interpolated_images)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

class DCGAN(Model):
    def __init__(self, generator, discriminator, latent_dim=input_dim):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            fake_images = self.generator(random_noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            d_loss_real = tf.reduce_mean(real_output)
            d_loss_fake = tf.reduce_mean(fake_output)
            gp = gradient_penalty(self.discriminator, real_images, fake_images)
            d_loss = d_loss_fake - d_loss_real + 10.0 * gp

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(random_noise, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_output)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {'d_loss': d_loss, 'g_loss': g_loss}

dcgan = DCGAN(generator=generator, discriminator=discriminator, latent_dim=input_dim)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

epochs = 40
lr_g = 0.0003
lr_d = 0.0001
beta = 0.5

dcgan.compile(
    g_optimizer=Adam(learning_rate=lr_g, beta_1=beta),
    d_optimizer=Adam(learning_rate=lr_d, beta_1=beta),
    loss_fn=BinaryCrossentropy()
)

import io
from tensorflow.keras.callbacks import Callback
import matplotlib
matplotlib.use('Agg')
class GANMonitor(Callback):
    def __init__(self, num_images=16, latent_dim=300, log_dir="logs"):
        self.num_images = num_images
        self.latent_dim = latent_dim
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        random_noise = tf.random.normal(shape=(self.num_images, self.latent_dim))
        generated_images = self.model.generator(random_noise, training=False)
        generated_images = (generated_images + 1) / 2  # Rescale to [0, 1] for display

        # Create a grid of generated images
        grid_image = self._create_image_grid(generated_images)

        # Write the combined image to TensorBoard
        with self.file_writer.as_default():
            tf.summary.image(f"Generated Images Epoch {epoch+1}", grid_image, step=epoch)

            # Log losses
            tf.summary.scalar("Generator Loss", logs['g_loss'], step=epoch)
            tf.summary.scalar("Discriminator Loss", logs['d_loss'], step=epoch)

    def _create_image_grid(self, images, rows=4, cols=4):
        """Creates a grid of images using matplotlib."""
        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < images.shape[0]:
                    img = images[idx].numpy()
                    axs[i, j].imshow(img)
                axs[i, j].axis('off')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        image = tf.image.decode_png(buf.getvalue(), channels=3)
        image = tf.expand_dims(image, 0) 
        return image
    

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
gan_monitor = GANMonitor(num_images=16, latent_dim=input_dim)

history = dcgan.fit(
    train_generator,
    epochs=epochs,
    callbacks=[tensorboard_callback, gan_monitor]
)


plt.figure(figsize=(10, 10))

for i in range(36):
    plt.subplot(6, 6, i + 1)
    # Noise
    noise = tf.random.normal([1, 300])
    mg = dcgan.generator(noise)
    mg = (mg * 255) + 255

    mg.numpy()
    image = Image.fromarray(np.uint8(mg[0]))

    plt.imshow(image)
    plt.axis('off')

plt.show()

import matplotlib.pyplot as plt

# Function to create a figure for the losses
def create_loss_figure(d_loss_values, g_loss_values):
    plt.figure(figsize=(10, 6))
    plt.plot(d_loss_values, label='Discriminator Loss')
    plt.plot(g_loss_values, label='Generator Loss')
    plt.title('Generator and Discriminator Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

d_loss_values = history.history['d_loss']
g_loss_values = history.history['g_loss']

create_loss_figure(d_loss_values, g_loss_values)

generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')