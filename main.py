import os
import tensorflow as tf
from openCV.ui import HandGUI

model = tf.keras.models.load_model("hand_cnn_model.keras")
dataset_path = "tensorFlow/dataset/train"
class_names = sorted(os.listdir(dataset_path))

print("Class names:", class_names)

app = HandGUI(model=model, class_names=class_names)
app.run()