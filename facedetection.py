import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

# Load model deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     'haarcascade_frontalface_default.xml')

# Use pipelines as high-level helpers
age_detection_pipeline = pipeline("image-classification", 
                                  model="dima806/facial_age_image_detection")
gender_detection_pipeline = pipeline("image-classification", 
                                     model="rizvandwiki/gender-classification")

# Inisialisasi video capture dari webcam
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()

    # Konversi frame ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Ekstrak ROI (Region of Interest) untuk wajah yang terdeteksi
        face_img = frame[y:y+h, x:x+w]

        # Convert ROI to RGB
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Convert to PIL image for pipelines
        pil_image = Image.fromarray(face_img_rgb)

        # Prediksi usia
        age_preds = age_detection_pipeline(pil_image)
        age = age_preds[0]['label']

        # Prediksi gender
        gender_preds = gender_detection_pipeline(pil_image)
        gender = gender_preds[0]['label']

        # Gambar kotak di sekitar wajah dan tulis usia dan Jenis Kelamin yang diprediksi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{age}, {gender}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Tampilkan hasil frame
    cv2.imshow('Deteksi Wajah, Jenis Kelamin dan Usia', frame)

    # Hentikan loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
