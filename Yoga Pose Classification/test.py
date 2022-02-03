import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import sys

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

from classification_models.tfkeras import Classifiers

from keras.callbacks import ModelCheckpoint



model_path=sys.argv[1] #"/kaggle/working/"
test_file=sys.argv[2] #"/kaggle/input/col341-a3/test.csv"
output_file=sys.argv[3] #"/kaggle/working/submission.csv"

testdf=pd.read_csv(test_file)
test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="./",
x_col="name",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(224,224))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

filepath = model_path+'best_resnet18.hdf5'

model = load_model(filepath)

preds = model.predict(test_generator)

predictions=np.argmax(preds,axis=1)

unencode = {0:'Ardhachakrasana',
 1:'Garudasana',
 2:'Gorakshasana',
 3:'Katichakrasana',
 4:'Natarajasana',
 5:'Natavarasana',
 6:'Naukasana',
 7:'Padahastasana',
 8:'ParivrittaTrikonasana',
 9:'Pranamasana',
 10:'Santolanasana',
 11:'Still',
 12:'Tadasana',
 13:'Trikonasana',
 14:'TriyakTadasana',
 15:'Tuladandasana',
 16:'Utkatasana',
 17:'Virabhadrasana',
 18:'Vrikshasana'}

df = pd.read_csv(test_file)
new_ans = []
for i in range(0,predictions.shape[0]):
    new_ans += [unencode[predictions[i]]]
predictions = new_ans[:-1]

data = {'name':df['name'].to_numpy()[:-1],
        'category': predictions}


df = pd.DataFrame(data)

df.to_csv(output_file,index = False)
