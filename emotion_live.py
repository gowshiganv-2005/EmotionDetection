
# Live Emotion detection with Age and Gender
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from deepface import DeepFace

# Load the pre-trained emotion model
model = load_model("emotion_model.h5")
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face for emotion detection (grayscale)
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_gray = cv2.resize(face_roi_gray, (48, 48))
        face_roi_gray = face_roi_gray / 255.0
        face_roi_gray = face_roi_gray.reshape(1, 48, 48, 1)

        # Predict Emotion
        pred = model.predict(face_roi_gray, verbose=0)
        emotion = labels[np.argmax(pred)]

        # Extract face for Age and Gender detection (color)
        face_roi_color = frame[y:y+h, x:x+w]
        
        try:
            # Predict Age and Gender using DeepFace
            results = DeepFace.analyze(face_roi_color, actions=['age', 'gender'], enforce_detection=False)
            age = int(results[0]['age'])
            gender = results[0]['dominant_gender']
            
            display_text = f"{emotion}, {gender}, {age}"
        except Exception as e:
            display_text = emotion

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, display_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Detection (Emotion, Gender, Age)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

