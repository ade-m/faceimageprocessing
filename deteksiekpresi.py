import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

# Load model deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     'haarcascade_frontalface_default.xml')

# Use pipelines as high-level helpers
expression_detection_pipeline = pipeline("image-classification", 
                                         model="trpakov/vit-face-expression")

# Warna untuk setiap ekspresi
expression_colors = {
    "happy": (0, 255, 0),       # Hijau
    "sad": (255, 0, 0),         # Biru
    "angry": (0, 0, 255),       # Merah
    "surprise": (255, 255, 0), # Kuning
    "disgust": (255, 100, 0),    # Biru
    "fear": (255, 0, 0),         # Biru
    "neutral": (255, 255, 255)  # Putih
}

# Inisialisasi video capture dari webcam
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()

    # Konversi frame ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                          minNeighbors=5, minSize=(50,50))

    for (x, y, w, h) in faces:
        # Ekstrak ROI (Region of Interest) untuk wajah yang terdeteksi
        face_img = frame[y:y+h, x:x+w]

        # Convert ROI to RGB
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Convert to PIL image for pipelines
        pil_image = Image.fromarray(face_img_rgb)

        # Prediksi ekspresi wajah
        expression_preds = expression_detection_pipeline(pil_image)
        expression = expression_preds[0]['label']

        # Gambar kotak di sekitar wajah dan tulis usia, gender, dan ekspresi yang diprediksi
        cv2.rectangle(frame, (x, y), (x+w, y+h), expression_colors[expression], 2)
        cv2.putText(frame, f"{expression}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    expression_colors[expression], 2)

    # Tampilkan hasil frame
    cv2.imshow('Deteksi Ekspresi Wajah', frame)

    # Hentikan loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
