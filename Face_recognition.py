import cv2
from deepface import DeepFace
import json
import os
from mtcnn import MTCNN
import threading
import queue

# Load pre-uploaded user data (face embeddings and associated details)
def load_user_data():
    user_data = {}
    for user_folder in os.listdir("uploads"):
        user_folder_path = os.path.join("uploads", user_folder)
        if os.path.isdir(user_folder_path):
            with open(os.path.join(user_folder_path, "info.json"), 'r') as f:
                data = json.load(f)
                user_data[user_folder] = data  # Store all data (including name, age, nationality)
    return user_data

# Compare embeddings to recognize face
def recognize_face(face_embedding, user_data):
    for name, data in user_data.items():
        embeddings = data['images']
        for stored_embedding in embeddings:
            # Use DeepFace.verify() to compare embeddings directly
            try:
                result = DeepFace.verify(face_embedding, stored_embedding, model_name='VGG-Face', enforce_detection=False)
                if result['verified']:
                    return data['name'], data['age'], data['nationality']
            except Exception as e:
                print(f"Error during face verification: {e}")
    return "Unknown", "N/A", "N/A"

# Function to process face and get embedding
def process_face(face_image, result_queue):
    try:
        embedding = DeepFace.represent(face_image, model_name="VGG-Face", enforce_detection=False)
        result_queue.put(embedding)  # Put result in the queue
    except Exception as e:
        print(f"Error during face embedding extraction: {e}")
        result_queue.put(None)

# Initialize MTCNN face detector and load user data
detector = MTCNN()
user_data = load_user_data()

# Open webcam
cap = cv2.VideoCapture(0)

# Create a queue to store results from threads
result_queue = queue.Queue()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Resize the frame for better performance
    frame = cv2.resize(frame, (640, 480))

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)
    print(f"Faces detected: {len(faces)}")  # Debug print

    threads = []  # List to store all threads for face processing

    for face in faces:
        x, y, w, h = face['box']
        
        # Crop the face from the frame
        face_image = frame[y:y+h, x:x+w]
        
        # Run face embedding extraction in a separate thread to avoid blocking
        embedding_thread = threading.Thread(target=process_face, args=(face_image, result_queue))
        threads.append(embedding_thread)  # Append the thread to the list
        embedding_thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Process the results from all threads
    for _ in range(len(faces)):
        embedding = result_queue.get()

        if embedding:
            recognized_name, recognized_age, recognized_nationality = recognize_face(embedding[0]['embedding'], user_data)

            # Draw bounding box and label the face with name, age, and nationality
            face = faces[_]  # Get the face corresponding to the thread
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Name: {recognized_name}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {recognized_age}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Nationality: {recognized_nationality}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
