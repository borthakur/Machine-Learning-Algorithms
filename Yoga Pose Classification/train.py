import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import sys

train_file=sys.argv[1] #"/kaggle/input/col341-a3/training.csv"
model_path=sys.argv[2] #"/kaggle/working/"

import tensorflow as tf

img_height = 299
img_width = 512

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
%matplotlib inline
from sklearn.utils import shuffle
import cv2

from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint

from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

traindf=pd.read_csv(train_file,dtype=str)
datagen=ImageDataGenerator(rescale=1./255,horizontal_flip=True,zoom_range=0.4,validation_split=0.2)

train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="./",
x_col="name",
y_col="category",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="./",
x_col="name",
y_col="category",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

from classification_models.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
base_model = ResNet18((224, 224, 3), weights='imagenet')

img_height,img_width = 224,224
num_classes = 19

x = base_model.layers[-3].output
x = Dense(512, activation= 'relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs = 12, validation_data=valid_generator)

adam = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

filepath = model_path+'best_resnet18.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]
history = model.fit(train_generator, epochs = 3, validation_data=valid_generator, callbacks=callbacks)

