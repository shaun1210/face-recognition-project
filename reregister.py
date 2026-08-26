"""Re-register existing students with fixed preprocessing pipeline."""
import os, sys, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import json
import numpy as np
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR   = os.path.join(BASE_DIR, 'db')
JSON     = os.path.join(BASE_DIR, 'students.json')

def preprocess(img):
    if img is None:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l = float(np.mean(l))
    clip = 2.5 if mean_l < 100 else 1.8
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    img_out = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    img_out = cv2.bilateralFilter(img_out, d=7, sigmaColor=50, sigmaSpace=50)
    if mean_l < 80:
        gamma = 1.3
        lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                        for i in range(256)], dtype=np.uint8)
        img_out = cv2.LUT(img_out, lut)
    return img_out

with open(JSON, 'r', encoding='utf-8') as f:
    students = json.load(f)

print(f"Students to re-register: {list(students.keys())}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    sys.exit(1)

print("Webcam opened. Taking photos in 3 seconds...")
import time
time.sleep(3)

for roll, info in students.items():
    ret, frame = cap.read()
    if not ret:
        print(f"  {roll}: Failed to capture frame")
        continue

    processed = preprocess(frame)
    path = os.path.join(DB_DIR, f'{roll}.jpg')
    cv2.imwrite(path, processed)
    print(f"  {roll} ({info['name']}): Saved new face image ({os.path.getsize(path)//1024}KB)")

cap.release()

# Verify with DeepFace
print("\nVerifying embeddings...")
for roll in students:
    path = os.path.join(DB_DIR, f'{roll}.jpg')
    try:
        emb = DeepFace.represent(
            img_path=path, model_name='ArcFace',
            detector_backend='retinaface', enforce_detection=False)
        vec = emb[0]['embedding']
        print(f"  {roll}: embedding OK (dim={len(vec)}, norm={np.linalg.norm(vec):.3f})")
    except Exception as e:
        print(f"  {roll}: FAILED - {e}")

print("\nDone! Re-register complete.")
