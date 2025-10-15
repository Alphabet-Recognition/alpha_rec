import random
import numpy as np

# Mock model that behaves like a Keras model
class MockModel:
    def predict(self, x):
        # Simulate prediction for a batch of 1 image with 26 classes (A-Z)
        num_classes = 26
        probs = np.random.rand(1, num_classes)
        probs = probs / probs.sum(axis=1, keepdims=True)  # normalize per-row
        return probs

# Example class names A-Z
class_names = [chr(ord("A") + i) for i in range(26)]

# Initialize GUI with mock model
from ui import HandGUI  # same-folder import

mock_model = MockModel()
gui = HandGUI(model=mock_model, class_names=class_names)
gui.run()