import cv2
import face_recognition
import os
import shutil
from pathlib import Path
import pickle
from datetime import datetime

# name = input("Enter the owner's name: ")

image_folder_path = str(f'training') #f'training/{name}'
# Make a timestamped directory to hold all these images
os.makedirs(f'{image_folder_path}', exist_ok=True)
# Make a directory for dumping encodings into
os.makedirs('output', exist_ok=True)

encodings_path = Path(f"output/encodings.pkl")

model = "hog" # hog mainly uses cpu, cnn mainly uses gpu

def encode_known_faces(model=model, encodings_location = encodings_path):
    names = []
    encodings = []
    for filepath in Path(image_folder_path).glob("*/*"): # Each subdirectory, "*" for just one folder
        print("Processing:", filepath)
        name = filepath.parent.name # Comment if you don't want it to accordingly name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations, model=model)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings} # Make a nice little array for all the data
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f) # Dump it all as a pickle file


encode_known_faces()
