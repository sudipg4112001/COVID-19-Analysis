# -*- coding: utf-8 -*-
"""coronahack-efficiantnetb2-f1-score-82.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X0mMbpdNtC8k0QOCgL8yhczjCjAnk6a4
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from pandas_profiling import ProfileReport

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image
# %matplotlib inline
import random
from tensorflow.keras.applications import EfficientNetB2

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

summary = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_dataset_Summary.csv')
metadata = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')
summary_profile = ProfileReport(summary, title="Summary DataFrame Profiling Report")
metadata_profile = ProfileReport(metadata, title="MetaData DataFrame Profiling Report")

metadata_profile

"""From metadata dataframe profile we can see there are 98.8% of labels missing for label_2_virus_cateogery, so we will implement classification on label_1_virus category i.e. bacteria vs virus.

> **Train Validation Test split:**
"""

test_images = list(metadata['X_ray_image_name'].loc[(metadata['Dataset_type'] == 'TEST') & (~metadata['Label_1_Virus_category'].isnull())])
train_images = list(metadata['X_ray_image_name'].loc[(metadata['Dataset_type'] == 'TRAIN') & (~metadata['Label_1_Virus_category'].isnull())])

train_virus_images = list(metadata['X_ray_image_name'].loc[(metadata['Dataset_type'] == 'TRAIN') &  (metadata['Label_1_Virus_category'] == 'Virus')])
train_bacteria_images = list(metadata['X_ray_image_name'].loc[(metadata['Dataset_type'] == 'TRAIN') &  (metadata['Label_1_Virus_category'] == 'bacteria')])


test_virus_images = list(metadata['X_ray_image_name'].loc[(metadata['Dataset_type'] == 'TEST') &  (metadata['Label_1_Virus_category'] == 'Virus')])
test_bacteria_images = list(metadata['X_ray_image_name'].loc[(metadata['Dataset_type'] == 'TEST') &  (metadata['Label_1_Virus_category'] == 'bacteria')])


print(f'No. of training images: {len(train_images)}')
print(f'No. of test images: {len(test_images)}')

print(f'No. of bacteria train images: {len(train_bacteria_images)}')
print(f'No. of virus train images: {len(train_virus_images)}')

print(f'No. of bacteria test images: {len(test_bacteria_images)}')
print(f'No. of virus test images: {len(test_virus_images)}')

#shuffle the data
train_bacteria_length = list(random.sample(range(0,2535),2535))

train_bacteria_paths = []
validation_bacteria_paths = []
for i,ele in enumerate(train_bacteria_length):
    if i<1267:
        train_bacteria_paths.append(f'../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/{train_bacteria_images[ele]}')
    else:
        validation_bacteria_paths.append(f'../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/{train_bacteria_images[ele]}')

#shuffle the data
train_virus_length = list(random.sample(range(0,1407),1407))

train_virus_paths = []
validation_virus_paths = []
for i,ele in enumerate(train_virus_length):
    if i<=703:
        train_virus_paths.append(f'../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/{train_virus_images[ele]}')
    else:
        validation_virus_paths.append(f'../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/{train_virus_images[ele]}')

test_virus_paths = []
test_bacteria_paths = []

for ele in test_virus_images:
    test_virus_paths.append(f'../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/{ele}')
    
for ele in test_bacteria_images:
    test_bacteria_paths.append(f'../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/{ele}')

"""The folder structure is created to load it into keras image data generator, here the classes are bacteria and virus.It is uploaded as bacteriavsvirus-trainvaltest-split zip folder. Classification is not possible at label 2 virus cateogry level due to less data (i.e less images with 2 virus cateogry label).

<h2> EfficientNet </h2>
EfficientNet, first introduced in Tan and Le, 2019 is among the most efficient models (i.e. requiring least FLOPS for inference) that reaches State-of-the-Art accuracy on both imagenet and common image classification transfer learning tasks.

<p>Compared to other models achieving similar ImageNet accuracy, EfficientNet is much smaller. For example, the ResNet50 model as you can see in Keras application has 23,534,592 parameters in total, and even though, it still underperforms the smallest EfficientNet, which only takes 5,330,564 parameters in total.</p>

> **Hyperparameters:**
"""

model = EfficientNetB2(weights='imagenet')
IMG_SIZE = 260
batch_size = 32
width = 260
height = 260
epochs = 10
dropout_rate = 0.2
input_shape = (height, width, 3)

# Commented out IPython magic to ensure Python compatibility.
# !git clone https://github.com/Tony607/efficientnet_keras_transfer_learning
# %cd '/kaggle/input/efficientnetkeras/efficientnet_keras_transfer_learning'

"""**Clone and import efficientNet** 

> Due to some error with tensorflow v2 , i corrected and imported the package seperately.

The EfficientNet is built for ImageNet classification contains 1000 classes labels. For our dataset, we only have 2. Which means the last few layers for classification is not useful for us. They can be excluded while loading the model by specifying the include_top argument to False, and this applies to other ImageNet models made available in Keras applications as well.
"""

from efficientnet import EfficientNetB2 as Net
from efficientnet import center_crop_and_resize, preprocess_input

conv_base = Net(weights="imagenet", include_top=False, input_shape=input_shape)
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
model.add(layers.Dense(2, activation="softmax", name="fc_out"))

# Commented out IPython magic to ensure Python compatibility.
# %cd '/kaggle/input/bacteriavsvirus-trainvaltest-split/bacteria_vs_virus'

train_dir= '/kaggle/input/bacteriavsvirus-trainvaltest-split/bacteria_vs_virus/train/'
validation_dir= '/kaggle/input/bacteriavsvirus-trainvaltest-split/bacteria_vs_virus/validation/'
test_dir = '/kaggle/input/bacteriavsvirus-trainvaltest-split/bacteria_vs_virus/test/'
NUM_TRAIN= 3942
NUM_TEST= 390

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode="categorical",
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode="categorical",
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode="categorical",
)

"""Here instead of freezing all layers and training with only top layer, i have unfreezed top 10 layers except batchnorm layers with smaller learning rate."""

def unfreeze_model(model):
    for layer in model.layers[-10:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

"""uncomment the following code to run training. I have uploaded the final checkpoint files in input after 20 epochs. Lets plot train vs validation accuracy and train vs validation loss."""

# # Include the epoch in the file name (uses `str.format`)
# checkpoint_path = "/kaggle/working/training_checkpoints/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights every 5 epochs
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, 
#     verbose=1, 
#     save_weights_only=True,
#     save_freq=10*batch_size)


# # Save the weights using the `checkpoint_path` format
# model.save_weights(checkpoint_path.format(epoch=0))


# unfreeze_model(model)

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=NUM_TRAIN // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=NUM_TEST // batch_size,
#     verbose=1,
#     use_multiprocessing=True,
#     workers=4,
#     callbacks=[cp_callback]
# )

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('/kaggle/input/coronohack-checkpoints/training_checkpoints/training_validation_acc.png')
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('/kaggle/input/coronohack-checkpoints/training_checkpoints//training_validation_loss.png')
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()

"""> From above graphs we can see that the model is starting to go into overfitting for higher epochs.

Lets find the latest checkpoint and load it into model. we will find the F-score on test data by finding probabilites using predict generator.
"""

conv_base = Net(weights="imagenet", include_top=False, input_shape=input_shape)
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
model.add(layers.Dense(2, activation="softmax", name="fc_out"))
latest = tf.train.latest_checkpoint('/kaggle/input/coronohack-checkpoints/training_checkpoints')
model.load_weights(latest)

test_generator = test_datagen.flow_from_directory(
        '/kaggle/input/bacteriavsvirus-trainvaltest-split/bacteria_vs_virus/test',
         target_size=(height, width),
         batch_size=batch_size,
         classes=['bacteria','virus'],
         class_mode='categorical',  
         shuffle=False)  

probabilities = model.predict_generator(test_generator)

from sklearn.metrics import confusion_matrix,classification_report

y_pred = np.argmax(probabilities, axis=1)

true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())   


report = classification_report(true_classes, y_pred, target_names=class_labels)
print(report)

"""<center><h2><b>F1 score achieved after 20 epochs and trying different hyperparameters is 82% </b><h2></center>

<center><h2><b>Thanks for reading. Please write your comments below.</b></h2></center>

<center><b>My References:</b> <a href="https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/">https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/</a> <br>
&emsp;&ensp;<a href="https://www.dlology.com/blog/transfer-learning-with-efficientnet/">https://www.dlology.com/blog/transfer-learning-with-efficientnet/</a>
</center>
"""