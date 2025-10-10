import os
import cv2
import numpy as np
import random
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import skimage.transform
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

imageSize = 96
train_dir = "tensorflow/dataset/train"
test_dir  = "tensorflow/dataset/test" 

def get_data(folder):
    X, y = [], []
    label_map = {
        'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9,
        'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,
        'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25,
        'del':26,'nothing':27,'space':28
    }

    for folderName in os.listdir(folder):
        path = os.path.join(folder, folderName)
        if not os.path.isdir(path):
            continue
        label = label_map.get(folderName, 29)
        for image_filename in tqdm(os.listdir(path), desc=f"Loading {folderName}"):
            img_path = os.path.join(path, image_filename)
            img_file = cv2.imread(img_path)
            if img_file is not None:
                img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
                img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                X.append(np.asarray(img_file))
                y.append(label)
    return np.asarray(X), np.asarray(y)

X, y = get_data(train_dir)
print("Loaded:", X.shape, y.shape)

# Split, Shuffle, One-Hot Encode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_trainHot = to_categorical(y_train, num_classes=30)
y_testHot = to_categorical(y_test, num_classes=30)

X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=13)
X_test, y_testHot = shuffle(X_test, y_testHot, random_state=13)

# Model Preparation
pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
for layer in pretrained_model.layers:
    layer.trainable = False

x = Flatten()(pretrained_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(30, activation='softmax')(x)
model = Model(inputs=pretrained_model.input, outputs=output)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Training
class MetricsCheckpoint(Callback):
    def __init__(self, savepath):
        super().__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    MetricsCheckpoint('logs.npy')
]

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(X_train)

history = model.fit(
    datagen.flow(X_train, y_trainHot, batch_size=64),
    validation_data=(X_test, y_testHot),
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Evaluation
score = model.evaluate(X_test, y_testHot, verbose=0)
print("Test Accuracy:", score[1])

y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(y_pred, axis=1)
Y_true = np.argmax(y_testHot, axis=1)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(10,10))
plt.imshow(confusion_mtx, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()

model.save("asl_vgg16_model.keras")