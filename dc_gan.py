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


# load image dataset for trainnig without labels
def load_image_train(filename, batch_size):
	# load image with the preferred size
    
    pixels = tf_dataset(filename, batch_size)
    
    return pixels


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
    model.add(tf.keras.layers.Reshape([4,4,512]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    
    model.add(tf.keras.layers.Conv2DTranspose(256, (2,2),strides=(2,2),use_bias=False,padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.Conv2D(128, (1,1),strides=(2,2), use_bias=False, padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
  
    
    model.add(tf.keras.layers.Conv2DTranspose(128, (2,2),strides=(2,2),use_bias=False,padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.Conv2D(64, (1,1),strides=(2,2), use_bias=False, padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    
    
    
    model.add(tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2),use_bias=False,padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(tf.keras.layers.Conv2D(32, (1,1),strides=(2,2), use_bias=False, padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    
    
    model.add(tf.keras.layers.Conv2DTranspose(3, (1,1), strides=(2,2),use_bias=False,padding="same",kernel_initializer=WEIGHT_INIT,
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
    
    model.add(tf.keras.layers.Conv2D(64,(5,5),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(128,(5,5),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="tanh"))
    
    return model


# In[ ]:


def save_plot(examples, name_model, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i])  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
    filename = f"samples/generated_plot-{name_model}.png"
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
    
    
    def compile(self, g_optimizer, d_optimizer, filepath, loss_fn, resume=False):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
            
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


    def saved_model(self, gmodelpath, dmodelpath):
        self.generator.save(gmodelpath)
        self.discriminator.save(dmodelpath)

    def loaded_model(self, g_filepath, d_filepath):
        self.generator.load_weights(g_filepath)
        self.discriminator.load_weights(d_filepath)


# In[ ]:


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self,
                 g_model_path,
                 d_model_path,
                 logs_file,
                 name_model
                ):
        super(CustomSaver, self).__init__()
        self.g_model_path = g_model_path
        self.d_model_path = d_model_path
        self.logs_file = logs_file
        self.name_model = name_model
        self.epochs_list = []
        self.gen_loss_list = []
        self.disc_loss_list = []
        
    
    def on_train_begin(self, logs=None):
        if not hasattr(self, 'epoch'):
            self.epoch = []
            self.history = {}
            
    def on_train_end(self, logs=None):
        self.model.saved_model(self.g_model_path, self.d_model_path)
        
        self.plot_epoch_result(self.epochs_list, self.gen_loss_list, "Generator_Loss", self.name_model, "g")
        self.plot_epoch_result(self.epochs_list, self.disc_loss_list, "Discriminator_Loss", self.name_model, "r")
    
    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
#             print(k, v)
            self.history.setdefault(k, []).append(v)
        
        self.epochs_list.append(epoch)
        self.gen_loss_list.append(logs["gen_loss"])
        self.disc_loss_list.append(logs["disc_loss"])

        
        
        if (epoch + 1) % 15 == 0 or (epoch + 1) <= 15:
            self.model.saved_model(self.g_model_path, self.d_model_path)
            print('saved for epoch',epoch + 1)
            
    def plot_epoch_result(self, epochs, loss, name, model_name, colour):
        plt.plot(epochs, loss, colour, label=name)
    #     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
        plt.title(name)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(model_name+ '_'+name+'_epoch_result.png')
        plt.show()
        plt.clf()

        
def scheduler(epoch, lr):
    if epoch < 1500:
        return lr
    else:
        return lr * tf.keras.math.exp(-0.1)

def set_callbacks(name_model, logs_path, logs_file, path_gmodal, path_dmodal, steps):
    # create and use callback:
    
    saver_callback = CustomSaver(
        path_gmodal,
        path_dmodal,
        logs_file,
        name_model
    )
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='disc_loss', factor=0.2,
                              patience=7, min_lr=0.000001)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_path + name_model + "/" + datetime.now().strftime("%Y%m%d-%H%M%S"), 
        histogram_freq=1
    )
    

    callbacks = [
        saver_callback,
#         checkpoints_callback,
        tensorboard_callback,
#         lr_callback,
        # reduce_lr,
    ]
    return callbacks


# In[ ]:


def run_trainning(model, train_dataset,num_epochs, path_gmodal, path_dmodal, logs_path, logs_file, name_model, steps, resume=False):

    
    
    callbacks = set_callbacks(name_model, logs_path, logs_file, path_gmodal, path_dmodal, steps)
            
    model.fit(train_dataset, epochs=num_epochs, callbacks=callbacks)
    
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
    
    batch_size = 64
    num_epochs = 150
    latent_dim = 100
    name_model= str(IMG_H)+"_dc_gan_"+str(num_epochs)
    
    resume_trainning = False
    lr = 1e-5
    
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
    
    resunetgan = DCGAN(g_model, d_model, latent_dim, batch_size)
    
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    
    resunetgan.compile(g_optimizer, d_optimizer, logs_file, bce_loss_fn, resume_trainning)
    
    """ run trainning process """
    train_images = glob(train_images_path)
    train_images_dataset = load_image_train(train_images, batch_size)
    train_images_dataset = train_images_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    size_of_dataset = len(list(train_images_dataset)) * batch_size
    
    steps = int(size_of_dataset/batch_size)
    run_trainning(resunetgan, train_images_dataset, num_epochs, path_gmodal, path_dmodal, logs_path, logs_file, name_model, steps,resume=resume_trainning)
    
    testing(g_model, path_gmodal, latent_dim, name_model)


# In[ ]:




