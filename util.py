import os
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from deepface import DeepFace

DEEPFACE_MODEL       = 'ArcFace'
DEEPFACE_BACKEND     = 'retinaface'
DISTANCE_METRIC      = 'cosine'
CONFIDENCE_THRESHOLD = 0.45


def preprocess_face(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return img_bgr

    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    mean_l = float(np.mean(l))
    clip   = 2.5 if mean_l < 100 else 1.8
    clahe  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq   = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    img_out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    img_out = cv2.bilateralFilter(img_out, d=7, sigmaColor=50, sigmaSpace=50)

    if mean_l < 80:
        gamma  = 1.3
        lut    = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                           for i in range(256)], dtype=np.uint8)
        img_out = cv2.LUT(img_out, lut)

    return img_out


def multi_exposure(img_bgr: np.ndarray):
    yield img_bgr
    yield cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=15)
    yield cv2.convertScaleAbs(img_bgr, alpha=0.85, beta=-8)


def get_button(window, text, color, command, fg='white'):
    return tk.Button(
        window, text=text,
        activebackground="black", activeforeground="white",
        fg=fg, bg=color, command=command,
        height=2, width=20,
        font=('Helvetica bold', 20)
    )

def get_img_label(window):
    return tk.Label(window)

def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label

def get_entry_text(window):
    return tk.Text(window, height=2, width=15, font=("Arial", 32))

def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img: np.ndarray, db_path: str) -> str:
    for variant in multi_exposure(img):
        processed = preprocess_face(variant)

        try:
            results = DeepFace.find(
                img_path=processed,
                db_path=db_path,
                model_name=DEEPFACE_MODEL,
                detector_backend=DEEPFACE_BACKEND,
                distance_metric=DISTANCE_METRIC,
                enforce_detection=False,
            )

            if not results or results[0].empty:
                continue

            best     = results[0].iloc[0]
            dist_col = [c for c in best.index if 'distance' in c.lower()]
            distance = float(best[dist_col[0]]) if dist_col else 1.0

            if distance <= CONFIDENCE_THRESHOLD:
                name       = os.path.splitext(os.path.basename(best['identity']))[0]
                confidence = round((1 - distance) * 100, 1)
                print(f"[DeepFace] Matched '{name}' | "
                      f"dist={distance:.3f} | conf={confidence}%")
                return name

        except Exception as e:
            print(f"[DeepFace] Recognition failed: {e}")
            continue

    try:
        faces = DeepFace.extract_faces(
            img_path=preprocess_face(img),
            enforce_detection=True,
            detector_backend='retinaface'
        )
        if faces:
            return 'unknown_person'
    except Exception:
        pass

    return 'no_persons_found'
