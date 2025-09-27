import tensorflow as tf
from openCV.ui import HandGUI

model = tf.keras.models.load_model("hand_cnn_model.keras")

digits = [str(i) for i in range(10)]
letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
extras = ["nothing", "del", "space"]
class_names = digits + letters + extras

app = HandGUI(model=model, class_names=class_names)
app.run()