# -*- coding: utf-8 -*-
"""covid-analysis-i.ipynb

# **Load required libraries**
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing import image
from PIL import Image

"""# **Loading dataset**"""

meta=pd.read_csv("../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv")
print(f'Shape: {meta.shape}')

meta.sample(10)

summ=pd.read_csv("../input/coronahack-chest-xraydataset/Chest_xray_Corona_dataset_Summary.csv")
print(f'Shape: {summ.shape}')

summ.sample(7)

print(f'Label counts in metadata:\n\n{meta.Label.value_counts()}')

print(f'Label-1 counts:\n\n{meta.Label_1_Virus_category.value_counts()}')

print(f'Label-2 counts:\n\n{meta.Label_2_Virus_category.value_counts()}')

print(f'Null counts in metdata:\n\n{meta.isnull().sum()}')

train_df = meta[meta.Dataset_type == 'TRAIN'].reset_index(drop=True)
test_df = meta[meta.Dataset_type == 'TEST'].reset_index(drop=True)
assert train_df.shape[0] + test_df.shape[0] == meta.shape[0]
print(f'Train df shape: {train_df.shape}')
print(f'Test df shape: {test_df.shape}')

"""# **Counting NULL values**"""

print(f'Count of null values in train:\n{train_df.isnull().sum()}')
print(f'\nCount of null values in test:\n{test_df.isnull().sum()}')

"""# **Replacing NULL values with "NA"**"""

train_df = train_df.fillna('NA')
test_df = test_df.fillna('NA')

train_df.sample(5)

test_df.sample(5)

test_df[test_df.Label == 'Pnemonia'].sample(5)

train_img="../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train"
test_img="../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test"

"""# **Visualizing Dataset**"""

import matplotlib.pyplot as plt
col = ['Label', 'Label_1_Virus_category', 'Label_2_Virus_category']

sns.set_theme(style='darkgrid')
fig = plt.figure(figsize=(14, 14))
for i in range(3):
    ax = plt.subplot(2, 2, i + 1)
    ax = sns.countplot(x=col[i], data=train_df)
    ax.set_title(f'Number of each value in {col[i]} column')
fig.suptitle('Count value in train_df')
plt.show()

pnemonia_df = train_df[train_df.Label == 'Pnemonia']
print(f'Number of pnemonia in training dataset: { len(pnemonia_df) }')

pnemonia_without_unknow_df = pnemonia_df[pnemonia_df.Label_2_Virus_category != 'NA']
print(f'Number of pnemonia without unknow value in Label_2_Virus_category: { len(pnemonia_without_unknow_df) }')

fig = plt.figure(figsize=(15, 4))
for i in range(2):
    ax = plt.subplot(1, 2, i+1)
    ax = sns.countplot(x=col[i+1], data=pnemonia_without_unknow_df)
    ax.set_title(col[i+1])
fig.suptitle('Count value in pnemonia_without_unknow_df')
plt.show()

assert os.path.isdir(train_img) == True
sample_train_images = list(os.walk(train_img))[0][2][:8]
sample_train_images = list(map(lambda x: os.path.join(train_img, x), sample_train_images))
plt.figure(figsize = (17,17))
for iterator, filename in enumerate(sample_train_images):
    image = Image.open(filename)
    plt.subplot(4,2,iterator+1)
    plt.imshow(image,cmap='gray')

plt.tight_layout()

fig, ax = plt.subplots(4, 2, figsize=(17, 17))
normal_path = train_df[train_df['Label']=='Normal']['X_ray_image_name'].values

sample_normal_path = normal_path[:4]
sample_normal_path = list(map(lambda x: os.path.join(train_img, x), sample_normal_path))

for row, file in enumerate(sample_normal_path):
    image = plt.imread(file)
    ax[row, 0].imshow(image,cmap='gray')
    ax[row, 1].hist(image.ravel(), 256, [0,256])
    ax[row, 0].axis('off')
    if row == 0:
        ax[row, 0].set_title('Images')
        ax[row, 1].set_title('Histograms')
fig.suptitle('Label = NORMAL', size=16)
plt.show()

fig, ax = plt.subplots(4, 2, figsize=(17, 17))
covid_path = train_df[train_df['Label_2_Virus_category']=='COVID-19']['X_ray_image_name'].values

sample_covid_path = covid_path[:4]
sample_covid_path = list(map(lambda x: os.path.join(train_img, x), sample_covid_path))

for row, file in enumerate(sample_covid_path):
    image = plt.imread(file)
    ax[row, 0].imshow(image,cmap='gray')
    ax[row, 1].hist(image.ravel(), 256, [0,256])
    ax[row, 0].axis('off')
    if row == 0:
        ax[row, 0].set_title('Images')
        ax[row, 1].set_title('Histograms')
fig.suptitle('Label 2 Virus Category = COVID-19', size=16)
plt.show()

test_df['Label'].value_counts()

meta.Dataset_type.value_counts()

train_data = meta[meta['Dataset_type'] == 'TRAIN']
test_data = meta[meta['Dataset_type'] == 'TEST']

def create_directory():
    try:
        os.makedirs('../working/train/Pnemonia')
        os.makedirs('../working/train/Normal')
        os.makedirs('../working/test/Pnemonia')
        os.makedirs('../working/test/Normal')
    except:
        pass

create_directory()

import shutil
train_pnemonia = '../working/train/Pnemonia/'
source_train = "../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train"
move_train_pnemonia = train_data[ train_data['Label'] == 'Pnemonia']['X_ray_image_name'].values
for i in move_train_pnemonia:
    path = os.path.join(source_train,i)
    shutil.copy(path,train_pnemonia)
    
#Normal
train_normal = '../working/train/Normal/'
move_train_normal = train_data[train_data.Label == 'Normal']['X_ray_image_name'].values
for i in move_train_normal:
    path = os.path.join(source_train,i)
    shutil.copy(path,train_normal)

test_pnemonia = '../working/test/Pnemonia/'
source_test = "../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test"
move_test_pnemonia = test_data[test_data['Label'] == 'Pnemonia']['X_ray_image_name'].values
                               
for i in move_test_pnemonia:
    
    path2 = os.path.join(source_test, i)
    shutil.copy(path2, test_pnemonia)

test_normal = '../working/test/Normal/'
move_test_normal = test_data[test_data.Label == 'Normal']['X_ray_image_name'].values
for i in move_test_normal:
    path3 = os.path.join(source_test, i)
    shutil.copy(path3, test_normal)

train_datagen = ImageDataGenerator(rescale = 1/255, rotation_range = 0.2, 
                              zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1/255)

train_gen = train_datagen.flow_from_directory("../working/train/", target_size = (200,200),
                                             batch_size = 50, class_mode = 'binary', 
                                              subset= 'training')
valid_gen = train_datagen.flow_from_directory("../working/train/", target_size = (200,200),
                                             batch_size = 50, class_mode = 'binary', 
                                              subset= 'validation')
test_gen = test_datagen.flow_from_directory("../working/test/", target_size = (200,200),
                                             batch_size = 50, class_mode = 'binary')

print(train_gen.class_indices)

"""# **Model Fitting**"""

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3), activation= 'relu',
                                                          input_shape= (200,200,3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(32,(3,3), activation= 'relu'),
                                    
                                   tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3), activation= 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3), activation= 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                    
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(1, activation = 'relu'),
                                   tf.keras.layers.Dense(1,activation = 'sigmoid')])
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_gen, validation_data = valid_gen, epochs = 10, 
                    callbacks = [callbacks], verbose = 1)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = (model.predict(test_gen)>0.5).astype("int32")

y_test = test_gen.labels
print('Classification report:\n', classification_report(y_test, pred))
print('Accuracy score:\n', accuracy_score(y_test, pred))

pred_class= model.predict_classes(test_gen)
print('Classification report:\n', classification_report(y_test, pred_class))
print('Accuracy Score:\n', accuracy_score(y_test, pred_class))

"""# **Saving the model**"""

from keras.models import load_model
model.save('model_lung_x-ray.h5')
