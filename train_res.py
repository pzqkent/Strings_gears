

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os



from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation,Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input

def identity_block(x,nb_filter,kernel_size=3):
    k1,k2,k3 = nb_filter
    out = Convolution2D(k1,1,1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2,kernel_size,kernel_size,border_mode='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3,1,1)(out)
    out = BatchNormalization()(out)


    out = merge([out,x],mode='sum')
    out = Activation('relu')(out)
    return out


def conv_block(x,nb_filter,kernel_size=3):
    k1,k2,k3 = nb_filter

    out = Convolution2D(k1,1,1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = out = Convolution2D(k2,kernel_size,kernel_size)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3,1,1)(out)
    out = BatchNormalization()(out)

    x = Convolution2D(k3,1,1)(x)
    x = BatchNormalization()(x)

    out = merge([out,x],mode='sum')
    out = Activation('relu')(out)
    return out

inp = Input(shape=(3,224,224))
out = ZeroPadding2D((3,3))(inp)
out = Convolution2D(64,7,7,subsample=(2,2))(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = MaxPooling2D((3,3),strides=(2,2))(out)

out = conv_block(out,[64,64,256])
out = identity_block(out,[64,64,256])
out = identity_block(out,[64,64,256])

out = conv_block(out,[128,128,512])
out = identity_block(out,[128,128,512])
out = identity_block(out,[128,128,512])
out = identity_block(out,[128,128,512])

out = conv_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])

out = conv_block(out,[512,512,2048])
out = identity_block(out,[512,512,2048])
out = identity_block(out,[512,512,2048])

out = AveragePooling2D((7,7))(out)
out = Flatten()(out)
out = Dense(2,activation='softmax')(out)

model = Model(inp,out)




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	if image is None:
		continue
	image = cv2.resize(image, (224, 224))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "damaged" else 0
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
# model = LeNet.build(width=200, height=200, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=Adam,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])