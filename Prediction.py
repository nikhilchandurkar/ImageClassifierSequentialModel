import tensorflow as tf
from PIL import Image
import numpy as np
from tkinter import Tk, filedialog

# Load the model
model = tf.keras.models.load_model('model.h5')
print("Model loaded successfully!")

# Define class labels (replace with your actual labels)
class_labels = ['cat', 'dog', 'car', 'plane']  # Example class names

# Function to preprocess the image
def preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to open a file dialog for image selection
def select_image():
    Tk().withdraw()  # Hide the root Tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        print("No file selected. Exiting.")
        return None
    return file_path

# Main program
image_path = select_image()
if image_path:
    print(f"Selected image: {image_path}")
    # Preprocess the image
    input_data = preprocess_image(image_path, target_size=(224, 224))  # Adjust size for your model
    # Predict
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)
    # Output the predicted class
    print(f"Predicted class: {class_labels[predicted_class[0]]}")
