import tensorflow as tf
import numpy as np
import pathlib

train_dir = pathlib.Path("tensorFlow/hand_landmark_dataset/train")
test_dir = pathlib.Path("tensorFlow/hand_landmark_dataset/test")

class GaussianBlurLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def get_gaussian_kernel(self, channels):
        ax = tf.range(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, dtype=tf.float32)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2. * tf.cast(self.sigma, tf.float32)**2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.expand_dims(kernel, axis=-1)
        kernel = tf.expand_dims(kernel, axis=-1)
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        return tf.cast(kernel, tf.float32)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        channels = tf.shape(inputs)[-1]
        kernel = self.get_gaussian_kernel(channels)
        return tf.nn.depthwise_conv2d(inputs, kernel, strides=[1,1,1,1], padding='SAME')

img_height, img_width = 64, 64
batch_size = 32
USE_AUGMENTATION = True

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomBrightness(factor=0.1),
    tf.keras.layers.RandomContrast(factor=0.1),
    tf.keras.layers.GaussianNoise(0.05),
    GaussianBlurLayer(kernel_size=3, sigma=0.5),
    tf.keras.layers.RandomSaturation(factor=0.1),
    tf.keras.layers.RandomHue(factor=0.1),
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

counts = {p.name: len(list((train_dir / p.name).glob('*'))) for p in train_dir.iterdir() if p.is_dir()}
print("Per-class counts (train):", counts)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# normalization [-1,1]
normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
if USE_AUGMENTATION:
    train_ds = train_ds.map(lambda x, y: (normalization_layer(data_augmentation(x)), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
else:
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y),
                      num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# create CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
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
    monitor="val_loss", 
    patience=5,           
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# train
model.fit(train_ds, epochs=20, validation_data=test_ds, callbacks=[callback, reduce_lr])

# acc, loss
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# predict
for images, labels in test_ds: 
    preds = model.predict(images) 
    for i in range(len(labels)): 
        print(f"ข้อความ: {class_names[np.argmax(labels[i])]} | ผล: {class_names[np.argmax(preds[i])]}")
model.save("hand_cnn_model.keras")