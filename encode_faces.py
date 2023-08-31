import os
import pickle
import face_recognition

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"

known_encodings = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]
            known_encodings.append(encoding)
            known_names.append(name)
        else:
            print(f"No face found in {KNOWN_FACES_DIR}/{name}/{filename}")

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

#https://realpython.com/face-recognition-with-python/ (structure + encoding info)