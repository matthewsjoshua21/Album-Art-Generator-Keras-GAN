from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.constraints import maxnorm

#Declare sequential models
Generator = Sequential()
Discriminator = Sequential()
Model = Sequential()

#Create layers
convolution_layer = Conv2D(filters=32, kernel_size=(3, 3), input_shape=[32, 32, 3], activation='relu', padding='same',
                           kernel_constraint=maxnorm(3))
leaky_layer = LeakyReLU(alpha=0.2)
normalize = BatchNormalization()
flatten = Flatten()

#Build Generator
Generator.add(Dense(128*128, input_dim=100))
Generator.add(leaky_layer)
Generator.add(normalize)


#Build Discriminator
Discriminator.add(convolution_layer)
Discriminator.add(leaky_layer)
Discriminator.add(normalize)
Discriminator.add(Flatten)
Discriminator.add(Dense(1, activation='sigmoid'))

#Build GAN
Model.add(Generator)
Model.add(Discriminator)
Model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

#Possible Training Algorithm