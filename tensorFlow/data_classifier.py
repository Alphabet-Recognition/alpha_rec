import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pathlib

train_dir = pathlib.Path("tensorFlow/dataset/train")
test_dir = pathlib.Path("tensorFlow/dataset/test")

img_height, img_width = 64, 64
batch_size = 32

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),        
    tf.keras.layers.RandomZoom(0.1),    
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(factor=0.2),
    tf.keras.layers.RandomContrast(factor=0.2),    
    tf.keras.layers.GaussianNoise(0.1),
    tfa.layers.GaussianBlur2D(
        kernel_size=3, sigma=1.0, padding='VALID'
    ),
    tf.keras.layers.RandomSaturation(factor=0.3),
    tf.keras.layers.RandomHue(factor=0.2),
])

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",           
    label_mode="categorical",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# normalization [-1,1]
normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# create CNN
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", 
    patience=5,           
    restore_best_weights=True
)

# train
model.fit(train_ds, epochs=20, validation_data=test_ds, callbacks=[callback])

# acc, loss
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# predict
for images, labels in test_ds: 
    preds = model.predict(images) 
    for i in range(len(labels)): 
        print(f"ข้อความ: {class_names[np.argmax(labels[i])]} | ผล: {class_names[np.argmax(preds[i])]}")