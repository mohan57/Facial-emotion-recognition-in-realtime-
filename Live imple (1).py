import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

classifier = load_model('C:/Users/Srinivas/Downloads/model3_200ep_ivc.h5')
face_cascade = cv2.CascadeClassifier("C:/Users/Srinivas/Downloads/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (218, 165, 32), 7)
        roi_gray = cv2.resize(gray_img[y:y + w, x:x + h], (48, 48))
        img_pixels = np.expand_dims(image.img_to_array(roi_gray), axis=(-1, 0))
        img_pixels /= 255
        predictions = classifier.predict(img_pixels)
        emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        cv2.putText(test_img, emotions[np.argmax(predictions[0])], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

    cv2.imshow('FER', cv2.resize(test_img, (1000, 700)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
