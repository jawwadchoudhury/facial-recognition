import cv2
import face_recognition
import os
import shutil
from pathlib import Path
import pickle
from datetime import datetime
from time import sleep 

# With a fair bit of help from https://realpython.com/face-recognition-with-python/
# And https://medium.com/@sunnykumar1516/access-camera-and-display-image-using-python-and-opencv-7e4b5d54375b

name = input("Enter the owner's name: ")
training_quality = int(input("Enter the desired training quality: ")) # How many photos should be encoded during one training session

training_folder_path = str(f'training/{int(datetime.now().timestamp())}_{name}')
# Make a timestamped directory to hold all these images
os.makedirs(f'{training_folder_path}', exist_ok=True)
# Make a directory for dumping encodings into
os.makedirs('output', exist_ok=True)

images_saved = 0
last_capture_time = datetime.now()

encodings_path = Path("output/encodings.pkl")

model = "hog" # hog mainly uses cpu, cnn mainly uses gpu

def encode_known_faces(model=model, encodings_location = encodings_path):
    names = []
    encodings = []
    for filepath in Path(training_folder_path).glob("*"):
        print("Processing:", filepath)
        # name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations, model=model)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings} # Make a nice little array for all the data
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f) # Dump it all as a pickle file
    
    # Delete training image files
    shutil.rmtree('training')

# Simple camera interface, using the capture
capture = cv2.VideoCapture(0)

while images_saved < training_quality:
    success, frame = capture.read() # Success is a boolean, returning whether reading the capture worked or not, frame is an image.
    

    if not success:
        print("Capturing the frame didn't work, maybe try again..?")
        break

    cv2.imshow("Training", frame) # A live camera feed

    now = datetime.now()
    if (datetime.now() - last_capture_time).total_seconds() >= 1:
        img_path = f'{training_folder_path}/{name}_{int(datetime.now().timestamp())}.jpg'
        cv2.imwrite(img_path, frame)  # Save the current frame
        images_saved += 1 # Increment
        print(f"Image {images_saved}/{training_quality} saved at: {img_path}")
        last_capture_time = now # Set current time as last capture time

    if cv2.waitKey(1) & 0xFF == ord('q'): # Press q to exit
        break

    

capture.release()
cv2.destroyAllWindows()

encode_known_faces()
