import tensorflow as tf
from openCV.ui import HandGUI

model = tf.keras.models.load_model("hand_cnn_model.keras")

alpha = [chr(i) for i in range(ord('A'), ord('Z')+1)]
num = [chr(i) for i in range(ord('1'), ord('9')+1)]
class_names = alpha + num

app = HandGUI(model=model, class_names=class_names)
app.run()