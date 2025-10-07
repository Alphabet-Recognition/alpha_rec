import os
import cv2
import numpy as np
import random
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import skimage.transform
from glob import glob
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import sklearn.metrics

imageSize = 160
train_dir = "tensorflow/dataset/train/"

def get_data(folder):
    X = []
    y = []
    
    # Label mapping
    label_map = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14,
        'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
        'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24,
        'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
        'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
        'Z': 35, 'del': 36, 'nothing': 37, 'space': 38
    }
    
    for folderName in os.listdir(folder):
        if not folderName.startswith('.') and folderName in label_map:
            label = label_map[folderName]
            folder_path = os.path.join(folder, folderName)
            
            if not os.path.isdir(folder_path):
                continue
            
            print(f"Loading {folderName}...")
            for image_filename in tqdm(os.listdir(folder_path)):
                if image_filename.startswith('.'):
                    continue
                    
                img_path = os.path.join(folder_path, image_filename)
                img_file = cv2.imread(img_path)
                
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, label_map

print("Loading data from train directory...")
X_data, y_data, label_map = get_data(train_dir)

print(f"\nTotal images loaded: {X_data.shape[0]}")
print(f"Number of classes: {len(np.unique(y_data))}")

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print(f"\nTraining set: {X_train.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

# Determine number of classes
num_classes = len(label_map)
print(f"Number of classes: {num_classes}")

# Encode labels to hot vectors
y_trainHot = to_categorical(y_train, num_classes=num_classes)
y_testHot = to_categorical(y_test, num_classes=num_classes)

# Shuffle data
X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=13)
X_test, y_testHot = shuffle(X_test, y_testHot, random_state=13)

print(f"\nFinal training set shape: {X_train.shape}")
print(f"Final test set shape: {X_test.shape}")

# Visualization functions
def plotHistogram(a):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(a)
    plt.axis('off')
    histo = plt.subplot(1,2,2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(a[:,:,0].flatten(), bins=n_bins, lw=0, color='r', alpha=0.5)
    plt.hist(a[:,:,1].flatten(), bins=n_bins, lw=0, color='g', alpha=0.5)
    plt.hist(a[:,:,2].flatten(), bins=n_bins, lw=0, color='b', alpha=0.5)
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=6)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(history):
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./training_curves.png')
    plt.show()

# Create reverse mapping for display
map_characters = {v: k for k, v in label_map.items()}
print("\nLabel mapping:", map_characters)

# Visualize label distribution
import seaborn as sns
df = pd.DataFrame()
df["labels"] = y_train
lab = df['labels']

plt.figure(figsize=(15,6))
ax = sns.countplot(x=lab, order=sorted(lab.unique()))
plt.title('Training Data Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
# Add class names to x-axis
ax.set_xticklabels([map_characters[i] for i in sorted(lab.unique())])
plt.tight_layout()
plt.show()

# Calculate class weights for imbalanced data
class_weight_array = class_weight.compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = dict(enumerate(class_weight_array))

print("\nClass weights calculated for balanced training")

# Build and train model
def pretrainedNetwork(X_train, y_train, X_test, y_test, 
                      num_classes, num_epochs, class_weights, labels):
    
    # Load VGG16 without top layers
    base_model = VGG16(weights='imagenet', include_top=False, 
                       input_shape=(imageSize, imageSize, 3))
    
    # Add custom top layers
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    
    model.summary()
    
    # Define callbacks
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=5, 
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            verbose=1,
            min_lr=1e-7
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train, 
        epochs=num_epochs, 
        batch_size=32,
        class_weight=class_weights, 
        validation_data=(X_test, y_test), 
        verbose=1,
        callbacks=callbacks_list
    )
    
    # Evaluate model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest accuracy: {score[1]:.4f}')
    print(f'Test loss: {score[0]:.4f}')
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    Y_pred_classes = np.argmax(y_pred, axis=1) 
    Y_true = np.argmax(y_test, axis=1)
    
    # Classification report
    print('\nClassification Report:')
    print(sklearn.metrics.classification_report(
        Y_true, Y_pred_classes, 
        target_names=[labels[i] for i in sorted(labels.keys())]
    ))
    
    # Confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    
    # Plot results
    plot_learning_curve(history)
    plot_confusion_matrix(
        confusion_mtx, 
        classes=[labels[i] for i in sorted(labels.keys())]
    )
    plt.show()
    
    return model

# Train the model
print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

model = pretrainedNetwork(
    X_train, y_trainHot, 
    X_test, y_testHot,
    num_classes=num_classes, 
    num_epochs=20,
    class_weights=class_weight_dict,
    labels=map_characters
)

# Save the final model
model.save('asl_vgg16_model.keras')