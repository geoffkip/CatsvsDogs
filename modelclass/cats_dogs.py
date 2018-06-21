# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class cats_dogs_nn:
	@staticmethod
	def build_keras():
		model = Sequential()
		# Add convulation
		model.add(Conv2D(32, (3,3), input_shape = (50,50,1),activation = 'relu'))

		# Add pooling layer
		model.add(MaxPooling2D(pool_size = (2,2)))

		# Add second convulational layer
		model.add(Conv2D(32, (3,3), activation = 'relu'))
		model.add(MaxPooling2D(pool_size = (2,2)))

		# Flattening
		model.add(Flatten())

		# Full connection and output layer
		model.add(Dense(units = 128, activation= 'relu'))
		model.add(Dense(units= 1 , activation = 'sigmoid'))

		# Compile the CNN model
		model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
		return model

	@staticmethod
	def build_tf():
		convnet = input_data(shape=[None, 50, 50, 1], name='input')

		convnet = conv_2d(convnet, 32, 5, activation='relu')
		convnet = max_pool_2d(convnet, 5)

		convnet = conv_2d(convnet, 64, 5, activation='relu')
		convnet = max_pool_2d(convnet, 5)

		convnet = fully_connected(convnet, 1024, activation='relu')
		convnet = dropout(convnet, 0.8)

		convnet = fully_connected(convnet, 2, activation='softmax')
		convnet = regression(convnet, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')

		model = tflearn.DNN(convnet, tensorboard_dir='log')
		return model
