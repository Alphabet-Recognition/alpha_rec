import os
import cv2
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import skimage.transform
from glob import glob
from tqdm import tqdm
from PIL import Image
import random, math

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


imageSize = 96
train_dir = "tensorflow/dataset/train"

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

class MatrixTransformer:
    def __init__(self, horizontal_flip=True, rotation_range=10,
                 width_shift_range=0.1, height_shift_range=0.1,
                 zoom_range=0.1, seed=None):
        self.horizontal_flip = horizontal_flip
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def flow_with_weights(self, X, y=None, batch_size=32, shuffle=True, class_weight=None):
        n = X.shape[0]
        indices = np.arange(n)
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                X_batch = np.array([self._augment_image(X[i]) for i in batch_idx])
                if y is not None:
                    y_batch = y[batch_idx]
                    if class_weight is not None:
                        sample_weight = np.array([class_weight[np.argmax(y[i])] for i in batch_idx])
                        yield X_batch, y_batch, sample_weight
                    else:
                        yield X_batch, y_batch
                else:
                    yield X_batch

    def _augment_image(self, img_array):
        img = Image.fromarray((img_array*255).astype(np.uint8))
        w, h = img.size
        cx, cy = w/2.0, h/2.0
        def T(tx, ty):
            return np.array([[1.0,0.0,tx],
                             [0.0,1.0,ty],
                             [0.0,0.0,1.0]])

        M = np.eye(3, dtype=float)

        if self.horizontal_flip and random.random() < 0.5:
            F = np.array([[-1,0,0],
                          [0,1,0],
                          [0,0,1]])
            M = T(cx, cy) @ F @ T(-cx, -cy) @ M

        if self.rotation_range:
            angle = random.uniform(-self.rotation_range/2.0, self.rotation_range/2.0)
            rad = math.radians(angle)
            R = np.array([[math.cos(rad), -math.sin(rad),0],
                          [math.sin(rad), math.cos(rad),0],
                          [0,0,1]])
            M = T(cx,cy) @ R @ T(-cx,-cy) @ M

        tx = random.uniform(-self.width_shift_range, self.width_shift_range)*w if self.width_shift_range else 0
        ty = random.uniform(-self.height_shift_range, self.height_shift_range)*h if self.height_shift_range else 0
        if tx != 0 or ty != 0:
            M = T(tx, ty) @ M

        if self.zoom_range:
            z = random.uniform(1.0 - self.zoom_range, 1.0 + self.zoom_range)
            S = np.array([[z,0,0],[0,z,0],[0,0,1]])
            M = T(cx,cy) @ S @ T(-cx,-cy) @ M

        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return img_array
        a,b,c,d,e,f = M_inv[0,0], M_inv[0,1], M_inv[0,2], M_inv[1,0], M_inv[1,1], M_inv[1,2]
        img = img.transform((w,h), Image.AFFINE, (a,b,c,d,e,f), resample=Image.BILINEAR)
        return np.asarray(img)/255.0

    def flow(self, X, y=None, batch_size=32, shuffle=True):
        n = X.shape[0]
        indices = np.arange(n)
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                X_batch = np.array([self._augment_image(X[i]) for i in batch_idx])
                if y is not None:
                    y_batch = y[batch_idx]
                    yield X_batch, y_batch
                else:
                    yield X_batch

X, y = get_data(train_dir)
print("Loaded:", X.shape, y.shape)

# Split, Shuffle, One-Hot Encode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_trainHot = to_categorical(y_train, num_classes=30)
y_testHot = to_categorical(y_test, num_classes=30)

X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=13)
X_test, y_testHot = shuffle(X_test, y_testHot, random_state=13)


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

datagen = MatrixTransformer(
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    seed=42
)

history = model.fit(
    datagen.flow_with_weights(X_train, y_trainHot, batch_size=64, class_weight=class_weights),
    validation_data=(X_test, y_testHot),
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

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