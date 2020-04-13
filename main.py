from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, BatchNormalization, Reshape
from keras.optimizers import SGD, Adam
from keras.constraints import maxnorm
from keras.datasets import mnist
import numpy as np

# Currently training model to work with mnist dataset until album art is gathered
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (_, _) = mnist.load_data()

x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)
y_train = y_train.reshape(-1, 1)

# Adversarial ground truths
valid = np.ones((128, 1))
fake = np.zeros((128, 1))

# Declare sequential models
Generator = Sequential()
Discriminator = Sequential()
Model = Sequential()

# Create layers
convolution_layer = Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1))
transpose_layer = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')
leaky_layer = LeakyReLU(alpha=0.2)
normalize = BatchNormalization()
flatten = Flatten()

# Build Discriminator
Discriminator.add(convolution_layer)
Discriminator.add(leaky_layer)
Discriminator.add(Dropout(0.4))
Discriminator.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
Discriminator.add(leaky_layer)
Discriminator.add(normalize)
Discriminator.add(Flatten())
Discriminator.add(Dense(1, activation='sigmoid'))
Discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

# Build Generator
Generator.add(Dense(128 * 7 * 7, input_dim=100))
Generator.add(leaky_layer)
Generator.add(Reshape((7, 7, 128)))
Generator.add(transpose_layer)
Generator.add(leaky_layer)
Generator.add(transpose_layer)
Generator.add(leaky_layer)
#Generator.add(normalize)
Generator.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))

# Build GAN
Model.add(Generator)
Model.add(Discriminator)
Model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Train the entire model
Discriminator.fit(x=x_train, y=y_train, epochs=1, batch_size=32)
Model.fit(x=x_train, y=y_train, epochs=1, batch_size=32)

# Possible Training Algorithm
