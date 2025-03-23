from flask import Flask, render_template, request, jsonify
import os
import json
from deepface import DeepFace
import cv2
from mtcnn import MTCNN

app = Flask(__name__)

# Route for the homepage (upload form)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and store face encodings
@app.route('/upload', methods=['POST'])
def upload_images():
    name = request.form['name']
    age = request.form['age']
    nationality = request.form['nationality']
    images = request.files.getlist('images')  # Get the list of uploaded images
    
    # Store user data and generate face embeddings
    user_folder = os.path.join('uploads', name)
    os.makedirs(user_folder, exist_ok=True)
    
    user_info = {'name': name, 'age': age, 'nationality': nationality, 'images': []}
    
    # Process images and extract face embeddings
    for i, img in enumerate(images):
        img_path = os.path.join(user_folder, f"image_{i+1}.jpg")
        img.save(img_path)  # Save the image
        
        # Generate face embedding using DeepFace
        embeddings = DeepFace.represent(img_path, model_name="VGG-Face", enforce_detection=False)
        if embeddings:
            user_info['images'].append(embeddings[0]['embedding'])  # Store embeddings
        
    # Save the user's information along with embeddings
    with open(os.path.join(user_folder, 'info.json'), 'w') as f:
        json.dump(user_info, f)
    
    return jsonify(message="Images and details uploaded successfully")

if __name__ == "__main__":
    app.run(debug=True)
