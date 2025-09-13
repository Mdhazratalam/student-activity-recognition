from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "Hazrat@1234"

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and label encoder
model = load_model('my_love.keras')  # Update to the correct model path

# Load the label encoder
train_labels = pd.read_csv("Training_set.csv")
label_encoder = LabelEncoder()
label_encoder.fit(train_labels['label'])

# Define image size according to what the model expects (224x224 based on the error)
img_size = (224, 224)  # Update to (224, 224) to match the model's expected input size

# Function to prepare image for prediction
def prepare_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Save the uploaded file

        # Prepare the image and make a prediction
        prepared_image = prepare_image(filepath)
        predictions = model.predict(prepared_image)
        predicted_class_index = np.argmax(predictions[0])  # Get index of highest probability
        predicted_label = label_encoder.inverse_transform([predicted_class_index])  # Get label
        
        return render_template('result.html', label=predicted_label[0], image_path=filepath)

if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
