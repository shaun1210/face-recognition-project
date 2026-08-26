import os
import datetime
import csv
import io
import json
import logging
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from deepface import DeepFace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('attendance')

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_DIR        = os.path.join(BASE_DIR, 'db')
LOG_PATH      = os.path.join(BASE_DIR, 'log.csv')
STUDENTS_JSON = os.path.join(BASE_DIR, 'students.json')

DEEPFACE_MODEL       = 'ArcFace'
DEEPFACE_BACKEND     = 'retinaface'
DISTANCE_METRIC      = 'cosine'
CONFIDENCE_THRESHOLD = 0.45
CSV_HEADER = ['Roll No', 'Name', 'Division', 'Department', 'Timestamp', 'Status']

# ── Boot-time setup ──────────────────────────────────────────────────────
os.makedirs(DB_DIR, exist_ok=True)

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(CSV_HEADER)
    log.info('Created log.csv')

if not os.path.exists(STUDENTS_JSON):
    with open(STUDENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump({}, f)
    log.info('Created students.json')

try:
    with open(STUDENTS_JSON, 'r', encoding='utf-8') as f:
        student_count = len(json.load(f))
except Exception:
    student_count = 0
log.info(f'Loaded {student_count} registered students')


# ── Error handlers ───────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "details": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    log.error(f'Internal error: {e}')
    return jsonify({"error": "Internal server error"}), 500


# ── Face preprocessing (skin-tone aware) ─────────────────────────────────

def preprocess_face(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    lab      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    mean_l   = float(np.mean(l))

    clip  = 2.5 if mean_l < 100 else 1.8
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)

    img_out = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    img_out = cv2.bilateralFilter(img_out, d=7, sigmaColor=50, sigmaSpace=50)

    if mean_l < 80:
        gamma = 1.3
        lut   = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                          for i in range(256)], dtype=np.uint8)
        img_out = cv2.LUT(img_out, lut)

    return img_out


def exposure_variants(img_bgr: np.ndarray):
    yield img_bgr
    yield cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=15)
    yield cv2.convertScaleAbs(img_bgr, alpha=0.85, beta=-8)


# ── Data helpers ──────────────────────────────────────────────────────────

def load_students() -> dict:
    with open(STUDENTS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_students(data: dict):
    with open(STUDENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def decode_image(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def log_attendance(roll: str, name: str, division: str,
                   department: str, status: str = 'present'):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([roll, name, division, department, ts, status])
    log.info(f'Logged: {name} ({roll}) -> {status}')


def already_marked_today(roll: str, status: str) -> bool:
    today = datetime.date.today().isoformat()
    if not os.path.exists(LOG_PATH):
        return False
    with open(LOG_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if (row.get('Roll No', '').strip() == roll and
                    row.get('Status', '').strip() == status and
                    row.get('Timestamp', '').startswith(today)):
                return True
    return False


def purge_deepface_cache():
    count = 0
    for fname in os.listdir(DB_DIR):
        if fname.endswith('.pkl'):
            try:
                os.remove(os.path.join(DB_DIR, fname))
                count += 1
            except OSError:
                pass
    if count:
        log.info(f'Purged {count} DeepFace cache files')


# ── Recognition (multi-strategy + multi-exposure) ────────────────────────

def smart_recognize(img_bgr: np.ndarray, db_path: str):
    for variant in exposure_variants(img_bgr):
        processed = preprocess_face(variant)

        try:
            results = DeepFace.find(
                img_path=processed,
                db_path=db_path,
                model_name=DEEPFACE_MODEL,
                detector_backend=DEEPFACE_BACKEND,
                enforce_detection=False,
                distance_metric=DISTANCE_METRIC,
            )

            if results and len(results[0]) > 0:
                best     = results[0].iloc[0]
                dist_col = [c for c in best.index if 'distance' in c.lower()]
                distance = float(best[dist_col[0]]) if dist_col else 1.0

                if distance <= CONFIDENCE_THRESHOLD:
                    name = os.path.splitext(
                        os.path.basename(best['identity']))[0]
                    log.info(f'Matched: {name} (dist={distance:.3f})')
                    return name, distance

        except Exception as e:
            log.debug(f'Recognition attempt failed: {e}')
            continue

    return None, None


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":  "ok",
        "service": "DBIT Face Attendance API",
        "version": "2.0",
        "students": len(load_students()),
        "db_dir": DB_DIR
    })


@app.route('/register', methods=['POST'])
def register():
    for field in ['roll', 'name', 'division', 'department']:
        if field not in request.form:
            return jsonify({"error": f"Missing field: {field}"}), 400
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    roll       = request.form['roll'].strip().upper()
    name       = request.form['name'].strip()
    division   = request.form['division'].strip().upper()
    department = request.form['department'].strip()

    if not roll or not name:
        return jsonify({"error": "Roll and name cannot be empty"}), 400

    file       = request.files['image']
    image_data = file.read()

    img_arr = np.frombuffer(image_data, np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    detected = False
    try:
        faces = DeepFace.extract_faces(
            img_path=preprocess_face(img_bgr),
            enforce_detection=True,
            detector_backend='retinaface')
        if faces:
            detected = True
    except Exception:
        pass

    if not detected:
        return jsonify({
            "error": "No face detected. Ensure good lighting and face the camera directly."
        }), 400

    preprocessed = preprocess_face(img_bgr)
    local_path   = os.path.join(DB_DIR, f"{roll}.jpg")
    cv2.imwrite(local_path, preprocessed)

    students       = load_students()
    students[roll] = {"roll": roll, "name": name,
                      "division": division, "department": department}
    save_students(students)

    purge_deepface_cache()

    log.info(f'Registered: {name} (Roll: {roll})')
    return jsonify({
        "status":  "success",
        "message": f"{name} (Roll: {roll}) registered successfully!"
    })


@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_bgr          = decode_image(request.files['image'])
    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    name_key, distance = smart_recognize(img_bgr, DB_DIR)

    if name_key is None:
        return jsonify({
            "status":  "unknown",
            "message": "Face not recognised. Please register or improve lighting."
        })

    students   = load_students()
    student    = students.get(name_key, {})
    roll       = student.get('roll',       name_key)
    full_name  = student.get('name',       name_key)
    division   = student.get('division',   'N/A')
    department = student.get('department', 'N/A')
    confidence = round((1 - distance) * 100, 1)

    duplicate = already_marked_today(roll, 'present')
    if not duplicate:
        log_attendance(roll, full_name, division, department, 'present')

    return jsonify({
        "status":     "success",
        "name":       full_name,
        "roll":       roll,
        "division":   division,
        "department": department,
        "confidence": confidence,
        "duplicate":  duplicate,
        "message":    (
            f"Welcome {full_name}! Attendance already marked today."
            if duplicate else
            f"Welcome {full_name}! Attendance marked. ({confidence}% confidence)"
        )
    })


@app.route('/exit', methods=['POST'])
def mark_exit():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_bgr = decode_image(request.files['image'])
    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    name_key, distance = smart_recognize(img_bgr, DB_DIR)

    if name_key is None:
        return jsonify({
            "status":  "unknown",
            "message": "Face not recognised."
        })

    students   = load_students()
    student    = students.get(name_key, {})
    roll       = student.get('roll',       name_key)
    full_name  = student.get('name',       name_key)
    division   = student.get('division',   'N/A')
    department = student.get('department', 'N/A')
    confidence = round((1 - distance) * 100, 1)

    duplicate = already_marked_today(roll, 'exit')
    if not duplicate:
        log_attendance(roll, full_name, division, department, 'exit')

    return jsonify({
        "status":     "success",
        "name":       full_name,
        "roll":       roll,
        "division":   division,
        "department": department,
        "confidence": confidence,
        "duplicate":  duplicate,
        "message":    (
            f"Goodbye {full_name}! Exit already marked today."
            if duplicate else
            f"Goodbye {full_name}! Exit marked. ({confidence}% confidence)"
        )
    })


@app.route('/export_csv', methods=['GET'])
def export_csv():
    date_filter = request.args.get('date', None)
    rows        = []

    with open(LOG_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if date_filter and len(row) >= 5:
                if not row[4].startswith(date_filter):
                    continue
            rows.append(row)

    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    buf.seek(0)

    filename = f"attendance_{date_filter or datetime.date.today()}.csv"
    return send_file(
        io.BytesIO(buf.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )


@app.route('/students', methods=['GET'])
def list_students():
    return jsonify(load_students())


@app.route('/attendance_today', methods=['GET'])
def attendance_today():
    today   = datetime.date.today().isoformat()
    records = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('Timestamp', '').startswith(today):
                    records.append(row)
    return jsonify(records)


if __name__ == '__main__':
    log.info('Starting DBIT Face Attendance API on port 5000')
    app.run(debug=True, host='0.0.0.0', port=5000)
