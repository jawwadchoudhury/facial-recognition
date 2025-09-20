import cv2
import face_recognition
from pathlib import Path
import pickle
from collections import Counter

encodings_path = Path("output/encodings.pkl") # File where face encodings are saved from training
model = "hog" # hog mainly uses cpu, cnn mainly uses gpu

def load_encodings(encodings_path):
    with encodings_path.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    encodings = loaded_encodings["encodings"]
    names = loaded_encodings["names"]
    return encodings, names

def get_face_locations(frame):
    face_locations = face_recognition.face_locations(
        frame, model=model
    )
    return face_locations

def get_face_encodings(frame, face_locations): 
    face_encodings = face_recognition.face_encodings(
        frame, face_locations, model=model
    )
    return face_encodings

def _recognize_face(frame_encoding, encodings, names):
    boolean_matches = face_recognition.compare_faces(
        encodings, frame_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, names)
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    
encodings, names = load_encodings(encodings_path)

capture = cv2.VideoCapture(0)
while True:
    success, frame = capture.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = get_face_locations(rgb_frame)
    face_encodings = get_face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        name = _recognize_face(face_encoding, encodings, names)
        if not name:
            name = "Unknown"
        
        y1, x2, y2, x1 = face_location
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
    

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
