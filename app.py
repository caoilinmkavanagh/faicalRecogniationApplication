import os
import pickle
import threading
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import face_recognition
import concurrent.futures


app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")
camera_lock = threading.Lock()

with open("encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

print("Loaded known names:")
print(known_names)


def process_face(face_location, face_encoding, frame):
    top, right, bottom, left = face_location

    # Scales back up face locations
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    # Find the closest match
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)

    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_names[first_match_index]

    # Draw a box around the face and labels it
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    return frame

def recognise_faces(frame):
    # Resizes frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converts to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find the face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Use ThreadPoolExecutor to process faces
    with concurrent.futures.ThreadPoolExecutor() as executor:
        frames = list(executor.map(process_face, face_locations, face_encodings, [frame] * len(face_locations)))

    return frames[0] if frames else frame

@app.route("/")
def index():
    return render_template("index.html")

def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        with camera_lock:
            ret, frame = cap.read()

        if not ret:
            break

        frame = recognise_faces(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001)


#References
#https://realpython.com/face-recognition-with-python/
#https://github.com/karenlo08/Face-Detector-Smart-Lock
#https://tutorialedge.net/python/intro-face-recognition-in-python/?source=post_page--------------------------- (encoding)
#https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html (colours - font etc)
#https://blog.miguelgrinberg.com/post/video-streaming-with-flask
#https://stackoverflow.com/questions/60009291/python-face-recognition-dataset-quality