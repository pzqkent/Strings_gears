from keras.models import Sequential
# from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D,Convolution2D
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.core import Dense
from keras import backend as K

# from keras.models import Sequential
# from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


class VGG_16:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)


		# model.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150)))
		model.add(Convolution2D(32, 3, 3, input_shape=inputShape))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(32, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64))
		model.add(Activation('relu'))

		model.add(Dropout(0.5))
		model.add(Dense(2))
		model.add(Activation('softmax'))





        
        

        
        
        

        
        



        
        
        

        
        
        

		# first set of CONV => RELU => POOL layers
		# model.add(Conv2D(20, (5, 5), padding="same",
		# 	input_shape=inputShape))
		# model.add(Activation("relu"))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# # second set of CONV => RELU => POOL layers
		# model.add(Conv2D(50, (5, 5), padding="same"))
		# model.add(Activation("relu"))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# # first (and only) set of FC => RELU layers
		# model.add(Flatten())
		# model.add(Dense(500))
		# model.add(Activation("relu"))

		# # softmax classifier
		# model.add(Dense(classes))
		# model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
