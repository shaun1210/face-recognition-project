"""Diagnostic script to test face recognition pipeline."""
import os, sys, io, codecs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
from deepface import DeepFace

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db')

print("=" * 60)
print("FACE RECOGNITION DIAGNOSTIC")
print("=" * 60)

# 1. Check registered faces
print("\n[1] Registered face images:")
for f in os.listdir(DB_DIR):
    if f.endswith('.jpg'):
        path = os.path.join(DB_DIR, f)
        img = cv2.imread(path)
        size = os.path.getsize(path)
        print(f"  {f}: {size//1024}KB, shape={img.shape if img is not None else 'NONE'}")

# 2. Test OpenCV face detection
print("\n[2] OpenCV detection on registered images:")
for f in os.listdir(DB_DIR):
    if f.endswith('.jpg'):
        path = os.path.join(DB_DIR, f)
        try:
            faces = DeepFace.extract_faces(
                img_path=path, detector_backend='opencv',
                enforce_detection=False)
            print(f"  {f}: OpenCV found {len(faces)} face(s)")
        except Exception as e:
            print(f"  {f}: OpenCV FAILED - {type(e).__name__}: {e}")

# 3. Test RetinaFace face detection
print("\n[3] RetinaFace detection on registered images:")
for f in os.listdir(DB_DIR):
    if f.endswith('.jpg'):
        path = os.path.join(DB_DIR, f)
        try:
            faces = DeepFace.extract_faces(
                img_path=path, detector_backend='retinaface',
                enforce_detection=False)
            print(f"  {f}: RetinaFace found {len(faces)} face(s)")
        except Exception as e:
            print(f"  {f}: RetinaFace FAILED - {type(e).__name__}: {e}")

# 4. Test embedding generation
print("\n[4] Embedding generation test:")
for f in os.listdir(DB_DIR):
    if f.endswith('.jpg'):
        path = os.path.join(DB_DIR, f)
        for backend in ['opencv', 'retinaface']:
            try:
                embedding = DeepFace.represent(
                    img_path=path, model_name='ArcFace',
                    detector_backend=backend, enforce_detection=False)
                vec = embedding[0]['embedding']
                print(f"  {f} ({backend}): embedding dim={len(vec)}, "
                      f"norm={np.linalg.norm(vec):.3f}")
            except Exception as e:
                print(f"  {f} ({backend}): FAILED - {type(e).__name__}: {e}")

# 5. Test face-to-face distance
print("\n[5] Cross-image distance test:")
imgs = [os.path.join(DB_DIR, f) for f in sorted(os.listdir(DB_DIR)) if f.endswith('.jpg')]
if len(imgs) >= 2:
    for backend in ['opencv', 'retinaface']:
        try:
            embs = []
            for img in imgs:
                emb = DeepFace.represent(
                    img_path=img, model_name='ArcFace',
                    detector_backend=backend, enforce_detection=False)
                embs.append(np.array(emb[0]['embedding']))

            from numpy.linalg import norm
            cos_dist = 1 - np.dot(embs[0], embs[1]) / (norm(embs[0]) * norm(embs[1]))
            conf = round((1 - cos_dist) * 100, 1)
            print(f"  {backend}: cosine_distance={cos_dist:.4f}, confidence={conf}%, "
                  f"threshold=0.45 -> {'PASS' if cos_dist <= 0.45 else 'FAIL'}")
        except Exception as e:
            print(f"  {backend}: FAILED - {type(e).__name__}: {e}")

# 6. Try find() with one image looking for the other
print("\n[6] DeepFace.find() test (cross-search):")
if len(imgs) >= 2:
    for backend in ['opencv', 'retinaface']:
        try:
            results = DeepFace.find(
                img_path=imgs[0], db_path=DB_DIR,
                model_name='ArcFace', detector_backend=backend,
                distance_metric='cosine', enforce_detection=False)
            if results and len(results[0]) > 0:
                best = results[0].iloc[0]
                dist_col = [c for c in best.index if 'distance' in c.lower()]
                dist = float(best[dist_col[0]]) if dist_col else -1
                matched = os.path.basename(best['identity'])
                print(f"  {backend}: matched={matched}, dist={dist:.4f}, "
                      f"conf={round((1-dist)*100,1)}% -> "
                      f"{'PASS' if dist <= 0.45 else 'FAIL (above threshold)'}")
            else:
                print(f"  {backend}: No results returned")
        except Exception as e:
            print(f"  {backend}: FAILED - {type(e).__name__}: {e}")

# 7. Capture webcam and test
print("\n[7] Live webcam test:")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        test_path = os.path.join(DB_DIR, '_test_webcam.jpg')
        cv2.imwrite(test_path, frame)
        print(f"  Captured frame: {frame.shape}")

        for backend in ['opencv', 'retinaface']:
            try:
                results = DeepFace.find(
                    img_path=frame, db_path=DB_DIR,
                    model_name='ArcFace', detector_backend=backend,
                    distance_metric='cosine', enforce_detection=False)
                if results and len(results[0]) > 0:
                    best = results[0].iloc[0]
                    dist_col = [c for c in best.index if 'distance' in c.lower()]
                    dist = float(best[dist_col[0]]) if dist_col else -1
                    matched = os.path.basename(best['identity'])
                    print(f"  {backend}: matched={matched}, dist={dist:.4f}, "
                          f"conf={round((1-dist)*100,1)}% -> "
                          f"{'PASS' if dist <= 0.45 else 'FAIL'}")
                else:
                    print(f"  {backend}: No match found")
            except Exception as e:
                print(f"  {backend}: FAILED - {type(e).__name__}: {e}")

        os.remove(test_path)
    cap.release()
else:
    print("  Cannot open webcam!")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
