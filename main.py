import os
import tensorflow as tf
from openCV.ui import HandGUI

model = tf.keras.models.load_model("asl_vgg16_model.keras")
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

print("Class names:", class_names)
app = HandGUI(model=model, class_names=class_names)
app.run()