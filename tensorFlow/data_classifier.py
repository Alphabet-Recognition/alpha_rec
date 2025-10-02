import tensorflow as tf
import numpy as np
import pathlib

train_dir = pathlib.Path("tensorFlow/dataset/train")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

img_height, img_width = 224, 224
batch_size = 32

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,  
    subset="training",
    seed=123,               
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

# Classnames
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Normalization
normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, 3, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Callbacks
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

# Train
history = model.fit(
    train_ds,
    epochs=40,
    validation_data=val_ds,
    callbacks=[callback, reduce_lr]
)

# Evaluate on validation set
val_loss, val_acc = model.evaluate(val_ds, verbose=2)
print(f"Validation accuracy: {val_acc:.4f}")

# Predict a few samples
for images, labels in val_ds.unbatch().take(200):
    preds = model.predict(tf.expand_dims(images, 0))
    print(f"จริง: {class_names[np.argmax(labels)]} | ทำนาย: {class_names[np.argmax(preds[0])]}")

# Save
model.save("hand_cnn_model.keras")