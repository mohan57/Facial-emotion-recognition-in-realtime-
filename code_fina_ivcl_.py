# -*- coding: utf-8 -*-
"""code_fina_ivcl .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1awE4faWcGqXyLeEfJjQQAHUYmvJt75m-
"""

import math
import numpy as np
import pandas as pd

import scikitplot
import seaborn as sns
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

from keras.utils import np_utils

from google.colab import drive
drive.mount('/content/drive')

!pip install scikit-plot

df = pd.read_csv('/content/drive/MyDrive/fer2013.csv')
print(df.shape)
df.head()
df.emotion.unique()

emotion_label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
df.emotion.value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='emotion', data=df)
plt.show()

def augment_pixels(px, IMG_SIZE=48):
    img = np.array(px.split()).reshape(IMG_SIZE, IMG_SIZE).astype('float32')
    img = tf.image.random_flip_left_right(img.reshape(IMG_SIZE, IMG_SIZE, 1))
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 12, IMG_SIZE + 12)
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 1])
    img = tf.clip_by_value(tf.image.random_brightness(img, 0.5), 0, 255)
    return ' '.join(img.numpy().reshape(IMG_SIZE * IMG_SIZE).astype('int').astype(str))

max_count = df.emotion.value_counts().max()
for emotion_idx, count_diff in (max_count - df.emotion.value_counts()).items():
    sampled = df.query("emotion == @emotion_idx").sample(count_diff, replace=True)
    sampled['pixels'] = sampled.pixels.apply(augment_pixels)
    df = pd.concat([df, sampled])
    print(emotion_idx, count_diff)

df.Usage.unique()

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='emotion', data=df)
plt.show()

import math
fig, axes = pyplot.subplots(7, 7, figsize=(14, 14))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
k = 0
for label in sorted(df.emotion.unique()):
    for j in range(7):
        px = np.array(df[df.emotion == label].pixels.iloc[k].split()).reshape(48, 48).astype('float32')
        k += 1
        ax = axes[label, j]
        ax.imshow(px, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(emotion_label_to_text[label], fontdict={'fontsize': 10})

pyplot.show()

df = df[df.emotion.isin([0, 1, 2, 3, 4, 5, 6])]
df.shape

img_array = np.stack(df.pixels.apply(lambda x: np.array(x.split()).reshape(48, 48, 1).astype('float32')), axis=0)
img_array.shape

img_labels = to_categorical(LabelEncoder().fit_transform(df.emotion))
print({cls: idx for idx, cls in enumerate(LabelEncoder().fit(df.emotion).classes_)})
#checking if the assignment is correct or wrong

(X_train, X_valid, y_train, y_valid) = train_test_split(img_array, img_labels, shuffle=True, stratify=img_labels, test_size=0.1, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]
X_train = X_train / 255.
X_valid = X_valid / 255.

from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

def build_net(optim):
    net = Sequential([
        Conv2D(64, (5, 5), input_shape=(img_width, img_height, img_depth), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(64, (5, 5), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        
        Flatten(),
        Dense(128, activation='elu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])
    
    net.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    net.summary()
    return net

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00005,
        patience=13,
        verbose=1,
        restore_best_weights=True,
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1,
    ),
]

# As the data in hand is less as compared to the task so ImageDataGenerator is good to go.
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
train_datagen.fit(X_train)

optims = [
    optimizers.Adam(0.001),
]
model = build_net(optims[0])

history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_valid, y_valid),
                    steps_per_epoch=len(X_train) / 32,
                    epochs=200,
                    callbacks=callbacks,
                    use_multiprocessing=True)

)

import pickle

with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save("/content/drive/MyDrive/Colab Notebooks/model3_200ep_ivc.h5")

from keras.models import load_model
model=load_model('/content/drive/MyDrive/Colab Notebooks/model3_200ep_ivc.h5')

import pickle

save_path = '/content/training_history.pkl'
with open(save_path, 'rb') as file:
    history = pickle.load(file)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(ax=axes[0], x=range(len(history['accuracy'])), y=history['accuracy'], label='train')
sns.lineplot(ax=axes[0], x=range(len(history['val_accuracy'])), y=history['val_accuracy'], label='valid')
axes[0].set_title('Accuracy')

sns.lineplot(ax=axes[1], x=range(len(history['loss'])), y=history['loss'], label='train')
sns.lineplot(ax=axes[1], x=range(len(history['val_loss'])), y=history['val_loss'], label='valid')
axes[1].set_title('Loss')

plt.tight_layout()
plt.savefig('epoch_history_dcnn.png')
plt.show()

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)

ax = pyplot.subplot(1, 2, 1)
sns.lineplot(range(len(history.history['accuracy'])), history.history['accuracy'], label='train')
sns.lineplot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'], label='valid')
pyplot.title('Accuracy')
pyplot.xlabel('Epoch')
pyplot.ylabel('Accuracy')

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(ax=axes[0], x=range(len(history['accuracy'])), y=history['accuracy'], label='train')
sns.lineplot(ax=axes[0], x=range(len(history['val_accuracy'])), y=history['val_accuracy'], label='valid')
axes[0].set_title('Accuracy')

sns.lineplot(ax=axes[1], x=range(len(history['loss'])), y=history['loss'], label='train')
sns.lineplot(ax=axes[1], x=range(len(history['val_loss'])), y=history['val_loss'], label='valid')
axes[1].set_title('Loss')

plt.tight_layout()
plt.savefig('epoch_history_dcnn.png')
plt.show()

import numpy as np
from sklearn.metrics import classification_report
import scikitplot
import matplotlib.pyplot as plt

predict_x = model.predict(X_valid)
classes_x = np.argmax(predict_x, axis=1)
y_valid_classes = np.argmax(y_valid, axis=1)

scikitplot.metrics.plot_confusion_matrix(y_valid_classes, classes_x, figsize=(7, 7))
plt.savefig("confusion_matrix_dcnn.png")

print(f'Number of Wrong Predictions: {np.sum(y_valid_classes != classes_x)}\n')
print(classification_report(y_valid_classes, classes_x))

mapper = {
    0: "disgust",
    1: "fear",
    2: "happiness",
    3: "sadness",
    4: "surprise",
    5: "sad",
    6: "neutral"
}

test_loss,test_acc=model.evaluate(X_valid,y_valid)
print("test loss ",test_loss)
print("test accuracy",test_acc)

predictions=model.predict([X_valid])

#Validation
import matplotlib.pyplot as plt

m=1333
prediction=np.argmax(predictions[m])
print("Predicted Emotion  :",mapper[prediction],":",prediction)


a=y_valid[m]

idx=0
for i in range (len(a)):
    if a[i]==1:
        idx=i
print("****************************")
print("True Emotion :",mapper[idx],":",idx)


my_array = np.squeeze(X_valid[m])
import matplotlib.pyplot as plt
plt.figure(figsize=(2, 2))
plt.imshow(my_array, cmap = "gray")

