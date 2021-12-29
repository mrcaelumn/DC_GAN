#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing Neccessary Library and constant variable

# !pip install tf_clahe
# !pip install -U scikit-learn
# !pip install matplotlib
# !pip install pandas


# In[ ]:


import itertools
import tensorflow as tf

import numpy as np
import pandas as pd 

from glob import glob
from tqdm import tqdm
from packaging import version
import os
from packaging import version
from datetime import datetime
# Import writer class from csv module
from csv import DictWriter

from matplotlib import pyplot as plt

IMG_H = 64
IMG_W = 64
IMG_C = 3  ## Change this to 1 for grayscale.

print("TensorFlow version: ", tf.keras.__version__)
assert version.parse(tf.keras.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."

# Weight initializers for the Generator network
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)
AUTOTUNE = tf.data.AUTOTUNE


# In[ ]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=IMG_C)
    img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)
    img = tf.cast(img, tf.float32)
#     rescailing image from 0,255 to -1,1
    img = (img - 127.5) / 127.5
    
    return img



def tf_dataset(images_path, batch_size, labels=False, class_names=None):
  
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


# In[ ]:


# we'll use cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    # First argument of loss is real labels
    # We've labeled our images as 1 (real) because
    # we're trying to fool discriminator
    return cross_entropy(tf.ones_like(fake_output),fake_output)


def discriminator_loss(real_images,fake_images):
    real_loss = cross_entropy(tf.ones_like(real_images),real_images)
    fake_loss = cross_entropy(tf.zeros_like(fake_images),fake_images)
    total_loss = real_loss + fake_loss
    return total_loss


# In[ ]:


# create generator model based on resnet50 and unet network
def build_generator(input_shape):
    model = tf.keras.Sequential()
    
    # Random noise to 16x16x256 image
    # model.add(tf.keras.layers.Dense(1024, activation="relu", use_bias=False, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(4*4*512, input_shape=input_shape))
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape([4,4,512]))
    
    
    model.add(tf.keras.layers.Conv2DTranspose(256, (5,5),strides=(2,2),use_bias=False,padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.Conv2D(128, (1,1),strides=(2,2), use_bias=False, padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
  
    
    model.add(tf.keras.layers.Conv2DTranspose(128, (5,5),strides=(2,2),use_bias=False,padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.Conv2D(64, (1,1),strides=(2,2), use_bias=False, padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    
    
    
    model.add(tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2),use_bias=False,padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.Conv2D(32, (1,1),strides=(2,2), use_bias=False, padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    
    
    model.add(tf.keras.layers.Conv2DTranspose(3, (5,5), strides=(2,2),use_bias=False,padding="same",kernel_initializer=WEIGHT_INIT,
                                     activation="tanh"
                                    ))
              # Tanh activation function compress values between -1 and 1. 
              # This is why we compressed our images between -1 and 1 in readImage function.
    # assert model.output_shape == (None,128,128,3)
    
    return model


# In[ ]:


# create discriminator model
def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(64,(5,5),strides=(2,2),padding="same", input_shape=input_shape))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(128,(5,5),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(256,(5,5),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(512,(5,5),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    
    return model


# In[ ]:


def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i])  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
    filename = f"samples/generated_plot_epoch-{epoch}.png"
    plt.savefig(filename)
    plt.close()
    

class DCGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator, latent_dim, batch_size):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
       
        # Regularization Rate for each loss function
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.999)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.999)
    
    
    def compile(self, g_optimizer, d_optimizer):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
            
# Notice the use of `tf.keras.function`
# This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        # We've created random seeds
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator generated images
            generated_images = self.generator(noise, training=True)

            # We've sent our real and fake images to the discriminator
            # and taken the decisions of it.
            real_output = self.discriminator(images,training=True)
            fake_output = self.discriminator(generated_images,training=True)

            # We've computed losses of generator and discriminator
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output,fake_output)

        # We've computed gradients of networks and updated variables using those gradients.
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss
        }


# In[ ]:


def testing(model, g_filepath, latent_dim , name_model, n_samples=25):
    noise = np.random.normal(size=(n_samples, latent_dim))

    # g_model = model.load(g_filepath)
    g_model = tf.keras.models.load_model(g_filepath)

    examples = g_model.predict(noise)
    save_plot(examples, name_model, int(np.sqrt(n_samples)))


# In[ ]:


if __name__ == "__main__":
    
    '''
    In Default:
    Clahe: OFF
    BCET: OFF
    Resize: crop or padding (decided by tensorflow)
    Datasets: For trainning dataset, it'll have additional datasets (flip-up-down and flip-right-left)
    '''
    
    # run the function here
    """ Set Hyperparameters """
    
    batch_size = 128
    num_epochs = 150
    latent_dim = 100
    name_model= str(IMG_H)+"_dc_gan_"+str(num_epochs)
    
    resume_trainning = False
    lr = 1e-4
    
    print("start: ", name_model)
    
    # set dir of files
    train_images_path = "data_test/*.jpg"
    saved_model_path = "saved_model/"
    
    logs_path = "logs/"
    
    logs_file = logs_path + "logs_" + name_model + ".csv"
    
    path_gmodal = saved_model_path + name_model + "_g_model" + ".h5"
    path_dmodal = saved_model_path +  name_model + "_d_model" + ".h5"
    
    """
    Create a MirroredStrategy object. 
    This will handle distribution and provide a context manager (MirroredStrategy.scope) 
    to build your model inside.
    """
    
    strategy = tf.distribute.MirroredStrategy()
    
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    input_shape = (IMG_H, IMG_W, IMG_C)
    # print(input_shape)
    
    ## init models ##
    
    d_model = build_discriminator(input_shape)
    g_model = build_generator((latent_dim, ))

    
#     d_model.summary()
#     g_model.summary()
    
    dcgan = DCGAN(g_model, d_model, latent_dim, batch_size)
    
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    
    dcgan.compile(g_optimizer, d_optimizer)
    
    """ run trainning process """
    train_images = glob(train_images_path)
    train_images_dataset = tf_dataset(train_images, batch_size)
    
    for epoch in range(num_epochs):
        epoch = epoch + 1
        print("epoch: ", epoch)
        dcgan.fit(train_images_dataset, epochs=1)
        if epoch % 25 == 0:
            print("saved at epoch: ", epoch)
            g_model.save(path_gmodal)
            d_model.save(path_dmodal)

            n_samples = 25
            noise = np.random.normal(size=(n_samples, latent_dim))
            examples = g_model.predict(noise)
            save_plot(examples, epoch, int(np.sqrt(n_samples)))
    
    # testing(g_model, path_gmodal, latent_dim, name_model)


# In[ ]:




