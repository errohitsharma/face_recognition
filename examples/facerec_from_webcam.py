import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"         # Folder containing known face images
UNKNOWN_FACES_DIR = "unknown_faces"     # Folder to save unknown face images
MODEL = "hog"                           # "hog" is fast, "cnn" is more accurate but slower
TOLERANCE = 0.6                         # Face matching tolerance (lower = stricter)

# Create folders if they don't exist
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Warning: '{KNOWN_FACES_DIR}' folder not found. Please create it and add face images.")
    os.makedirs(KNOWN_FACES_DIR)

os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Load known faces
print("Loading known faces...")

known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    
    if not os.path.isfile(filepath):
        continue

    try:
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            print(f"Warning: No face found in '{filename}'.")

    except Exception as e:
        print(f"Error loading '{filename}': {e}")

print(f"Loaded {len(known_face_names)} known faces: {known_face_names}")

# Start webcam
video_capture = cv2.VideoCapture(0)
print("Starting webcam...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR (OpenCV) to RGB (face_recognition)

    # Detect face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names_in_frame = []

    for face_encoding in face_encodings:
        name = "Unknown"

        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

        face_names_in_frame.append(name)

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names_in_frame):
        # Scale back up face locations since the frame was scaled down
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Label the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Save unknown faces
        if name == "Unknown":
            face_image = frame[top:bottom, left:right]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp}.jpg"
            save_path = os.path.join(UNKNOWN_FACES_DIR, filename)
            cv2.imwrite(save_path, face_image)

    # Show the video frame
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
print("Program terminated.")
