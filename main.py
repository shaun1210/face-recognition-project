import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import csv
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import util

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
STUDENTS_JSON = os.path.join(BASE_DIR, 'students.json')
DB_DIR        = os.path.join(BASE_DIR, 'db')
LOG_PATH      = os.path.join(BASE_DIR, 'log.csv')

CSV_HEADER = ['Roll No', 'Name', 'Division', 'Department', 'Timestamp', 'Status']

DEPARTMENTS = [
    "Computer Engineering",
    "Information Technology",
    "Electronics & Telecomm.",
    "Mechanical Engineering",
]
DIVISIONS = ["A", "B"]

# ── Theme colours ─────────────────────────────────────────────────────────
BG_DARK   = "#0f1119"
BG_PANEL  = "#161825"
FG_LIGHT  = "#e0e4ef"
FG_MUTED  = "#6b7394"
BTN_GREEN = "#22c55e"
BTN_RED   = "#ef4444"
BTN_BLUE  = "#5b7cf7"
BTN_AMBER = "#f59e0b"
BTN_PURPLE= "#8b5cf6"
ENTRY_BG  = "#1e2035"
BORDER    = "#252840"


# ── Data helpers ──────────────────────────────────────────────────────────

def load_students() -> dict:
    if os.path.exists(STUDENTS_JSON):
        with open(STUDENTS_JSON, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_students(data: dict):
    with open(STUDENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def init_log():
    os.makedirs(os.path.dirname(LOG_PATH) or '.', exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_log(roll: str, name: str, division: str, department: str, status: str):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([roll, name, division, department, ts, status])


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
    for fname in os.listdir(DB_DIR):
        if fname.endswith('.pkl'):
            try:
                os.remove(os.path.join(DB_DIR, fname))
            except OSError:
                pass


# ── Styled button helper ──────────────────────────────────────────────────

def make_button(parent, text, color, command, **extra):
    cfg = dict(
        text=text, bg=color, fg='white',
        activebackground='#1a1a2e', activeforeground='white',
        command=command, height=2, width=26,
        font=('Helvetica bold', 14), relief='flat',
        cursor='hand2', bd=0, highlightthickness=0,
    )
    cfg.update(extra)
    btn = tk.Button(parent, **cfg)
    btn.bind('<Enter>', lambda e: btn.configure(bg=color + 'dd'))
    btn.bind('<Leave>', lambda e: btn.configure(bg=color))
    return btn


# ── Main application ──────────────────────────────────────────────────────

class App:
    def __init__(self):
        init_log()
        os.makedirs(DB_DIR, exist_ok=True)

        self.root = tk.Tk()
        self.root.geometry("1220x580+300+60")
        self.root.title("DBIT Face Attendance System")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        self._build_ui()
        self._start_webcam()

    def _build_ui(self):
        # Left: live feed
        self.cam_label = tk.Label(self.root, bg='#000', bd=0)
        self.cam_label.place(x=10, y=10, width=700, height=520)

        # Right panel
        panel = tk.Frame(self.root, bg=BG_PANEL, bd=0, highlightthickness=1,
                         highlightbackground=BORDER)
        panel.place(x=720, y=10, width=490, height=560)

        # Title
        title_frame = tk.Frame(panel, bg=BG_PANEL)
        title_frame.place(x=20, y=16, width=450, height=40)

        tk.Label(title_frame, text="DBIT", bg=BG_PANEL, fg=BTN_BLUE,
                 font=("Helvetica bold", 22)).pack(side='left')
        tk.Label(title_frame, text=" Attendance", bg=BG_PANEL, fg=FG_LIGHT,
                 font=("Helvetica bold", 22)).pack(side='left')

        # Status box
        self.status_var = tk.StringVar(value="  System ready")
        status_frame = tk.Frame(panel, bg='#1a1d2e', bd=0,
                                highlightthickness=1, highlightbackground=BORDER)
        status_frame.place(x=20, y=58, width=450, height=44)

        self.status_label = tk.Label(status_frame, textvariable=self.status_var,
                                     bg='#1a1d2e', fg=FG_LIGHT,
                                     font=("Helvetica", 11), wraplength=420,
                                     justify='left', anchor='w')
        self.status_label.pack(fill='both', expand=True, padx=8)

        # Separator
        tk.Frame(panel, bg=BORDER, height=1, width=450).place(x=20, y=112)

        # Section label
        tk.Label(panel, text="ACTIONS", bg=BG_PANEL, fg=FG_MUTED,
                 font=("Helvetica", 9), anchor='w').place(x=20, y=122)

        # Action buttons
        buttons = [
            ("  Mark Attendance",  BTN_GREEN, self.login,             145),
            ("  Mark Exit",         BTN_RED,   self.logout,            220),
            ("  Register Student",  BTN_BLUE,  self.register_new_user, 295),
            ("  View Today's Log",  BTN_AMBER, self.show_log,          370),
        ]
        for text, colour, cmd, y in buttons:
            btn = make_button(panel, text, colour, cmd)
            btn.place(x=20, y=y)

        # Footer
        tk.Frame(panel, bg=BORDER, height=1, width=450).place(x=20, y=510)
        tk.Label(panel, text="All data stored locally on this device.",
                 bg=BG_PANEL, fg=FG_MUTED,
                 font=("Helvetica", 9)).place(x=20, y=525)

        # Version badge
        tk.Label(panel, text="v2.0", bg=BG_PANEL, fg=FG_MUTED,
                 font=("Helvetica", 8)).place(x=440, y=525)

    # ── Webcam ────────────────────────────────────────────────────────────

    def _start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.latest_frame = None
        self._tick()

    def _tick(self):
        ret, frame = self.cap.read()
        if ret:
            self.latest_frame = frame
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
        self.cam_label.after(20, self._tick)

    # ── Attendance actions ────────────────────────────────────────────────

    def _record(self, status: str, greeting: str):
        if self.latest_frame is None:
            util.msg_box("Error", "Webcam not ready.")
            return

        self.status_var.set("  Scanning face...")
        self.status_label.configure(fg=BTN_BLUE)
        self.root.update()

        key = util.recognize(self.latest_frame, DB_DIR)

        if key == 'no_persons_found':
            util.msg_box('No Face Detected',
                         'No face found in frame.\n'
                         'Ensure your face is well-lit and centred.')
            self.status_var.set("  No face detected.")
            self.status_label.configure(fg=BTN_RED)
            return

        if key == 'unknown_person':
            util.msg_box('Not Recognised',
                         'Face detected but not in the database.\n'
                         'Please register first.')
            self.status_var.set("  Unknown face detected.")
            self.status_label.configure(fg=BTN_RED)
            return

        students = load_students()
        s        = students.get(key, {})
        roll     = s.get('roll', key)
        name     = s.get('name', key)
        division = s.get('division', 'N/A')
        dept     = s.get('department', 'N/A')

        if already_marked_today(roll, status):
            util.msg_box('Already Recorded',
                         f'{name} ({roll}) is already marked "{status}" today.')
            self.status_var.set(f"  {name} already marked today.")
            self.status_label.configure(fg=BTN_AMBER)
            return

        append_log(roll, name, division, dept, status)

        util.msg_box(greeting, f'{greeting}, {name}!\nRoll: {roll}  |  Div: {division}')
        self.status_var.set(f"  {greeting}, {name} ({roll}) - logged to CSV.")
        self.status_label.configure(fg=BTN_GREEN)

    def login(self):
        self._record('present', 'Welcome')

    def logout(self):
        self._record('exit', 'Goodbye')

    # ── View today's log ─────────────────────────────────────────────────

    def show_log(self):
        today  = datetime.date.today().isoformat()
        rows   = []

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('Timestamp', '').startswith(today):
                        rows.append(row)

        win = tk.Toplevel(self.root)
        win.title(f"Attendance - {today}")
        win.geometry("900x420+300+200")
        win.configure(bg=BG_DARK)

        # Header
        hdr = tk.Frame(win, bg=BG_PANEL, height=50)
        hdr.pack(fill='x')
        tk.Label(hdr, text=f"  Today's Attendance  ({len(rows)} records)",
                 bg=BG_PANEL, fg=FG_LIGHT,
                 font=("Helvetica bold", 14)).pack(side='left', padx=10, pady=10)

        cols = CSV_HEADER
        tree = ttk.Treeview(win, columns=cols, show='headings', height=15)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=140, anchor='center')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview",
                         background=BG_PANEL, foreground=FG_LIGHT,
                         rowheight=30, fieldbackground=BG_PANEL,
                         font=('Helvetica', 11), borderwidth=0)
        style.configure("Treeview.Heading",
                         background=BG_DARK, foreground=BTN_BLUE,
                         font=('Helvetica bold', 11), borderwidth=0)
        style.map("Treeview", background=[('selected', '#252840')])

        for r in rows:
            tree.insert('', 'end', values=[r.get(c, '') for c in cols])

        sb = ttk.Scrollbar(win, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        sb.pack(side='right', fill='y', pady=10, padx=(0, 10))

        if not rows:
            tk.Label(win, text="No attendance records for today yet.",
                     bg=BG_DARK, fg=FG_MUTED,
                     font=('Helvetica', 14)).place(relx=0.5, rely=0.5, anchor='center')

    # ── Registration window ──────────────────────────────────────────────

    def register_new_user(self):
        if self.latest_frame is None:
            util.msg_box("Error", "Webcam not ready.")
            return

        snap = self.latest_frame.copy()

        win = tk.Toplevel(self.root)
        win.geometry("1220x580+320+100")
        win.title("Register New Student")
        win.configure(bg=BG_PANEL)

        # Show snapshot
        snap_label = tk.Label(win, bg='#000', bd=0)
        snap_label.place(x=10, y=10, width=700, height=520)
        rgb    = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
        imgtk  = ImageTk.PhotoImage(Image.fromarray(rgb))
        snap_label.imgtk = imgtk
        snap_label.configure(image=imgtk)

        # Form panel
        form_panel = tk.Frame(win, bg=BG_PANEL, bd=0,
                              highlightthickness=1, highlightbackground=BORDER)
        form_panel.place(x=720, y=10, width=490, height=560)

        tk.Label(form_panel, text="Register Student", bg=BG_PANEL, fg=FG_LIGHT,
                 font=("Helvetica bold", 18)).place(x=20, y=16)

        tk.Frame(form_panel, bg=BORDER, height=1, width=450).place(x=20, y=52)

        capture_holder = {'frame': snap}

        def retake():
            if self.latest_frame is not None:
                capture_holder['frame'] = self.latest_frame.copy()
                rgb2   = cv2.cvtColor(capture_holder['frame'], cv2.COLOR_BGR2RGB)
                imgtk2 = ImageTk.PhotoImage(Image.fromarray(rgb2))
                snap_label.imgtk = imgtk2
                snap_label.configure(image=imgtk2)
                status_lbl.config(text="  New photo captured.", fg=BTN_GREEN)

        lbl_cfg   = dict(bg=BG_PANEL, fg=FG_MUTED, font=('Helvetica', 10),
                         anchor='w')
        entry_cfg = dict(font=('Helvetica', 13), width=22, relief='flat',
                         bg=ENTRY_BG, fg='white', insertbackground='white',
                         bd=0, highlightthickness=1, highlightbackground=BORDER)

        fields = {}
        y_offset = 70
        for key, label_text in [('roll', 'Roll Number'), ('name', 'Full Name')]:
            tk.Label(form_panel, text=label_text, **lbl_cfg).place(x=20, y=y_offset)
            e = tk.Entry(form_panel, **entry_cfg)
            e.place(x=20, y=y_offset + 22, width=450, height=32)
            fields[key] = e
            y_offset += 68

        tk.Label(form_panel, text='Division', **lbl_cfg).place(x=20, y=y_offset)
        div_var = tk.StringVar(value=DIVISIONS[0])
        ttk.Combobox(form_panel, textvariable=div_var, values=DIVISIONS,
                     font=('Helvetica', 13), width=10,
                     state='readonly').place(x=20, y=y_offset + 22, width=200, height=32)
        y_offset += 68

        tk.Label(form_panel, text='Department', **lbl_cfg).place(x=20, y=y_offset)
        dept_var = tk.StringVar(value=DEPARTMENTS[0])
        ttk.Combobox(form_panel, textvariable=dept_var, values=DEPARTMENTS,
                     font=('Helvetica', 12), width=28,
                     state='readonly').place(x=20, y=y_offset + 22, width=450, height=32)
        y_offset += 68

        status_lbl = tk.Label(form_panel, text="", bg=BG_PANEL, fg=BTN_AMBER,
                               font=('Helvetica', 10), wraplength=400, anchor='w')
        status_lbl.place(x=20, y=y_offset)

        def accept():
            roll = fields['roll'].get().strip().upper()
            name = fields['name'].get().strip()
            div  = div_var.get().strip()
            dept = dept_var.get().strip()

            if not roll or not name:
                util.msg_box('Error', 'Roll number and name are required.')
                return

            img_path = os.path.join(DB_DIR, f'{roll}.jpg')
            face_img = util.preprocess_face(capture_holder['frame'])
            cv2.imwrite(img_path, face_img)

            students       = load_students()
            students[roll] = {'roll': roll, 'name': name,
                              'division': div, 'department': dept}
            save_students(students)

            purge_deepface_cache()

            util.msg_box('Registered!',
                         f'{name} ({roll}) registered.\n'
                         f'They will appear in the CSV the next time\n'
                         f'attendance is marked.')
            self.status_var.set(f"  Registered: {name} ({roll})")
            self.status_label.configure(fg=BTN_GREEN)
            win.destroy()

        btn_y = y_offset + 40
        make_button(form_panel, '  Retake Photo', BTN_PURPLE, retake,
                    width=20, font=('Helvetica bold', 12)).place(x=20, y=btn_y)

        make_button(form_panel, '  Confirm Registration', BTN_GREEN, accept,
                    width=26, font=('Helvetica bold', 13)).place(x=20, y=btn_y + 48)

        make_button(form_panel, '  Cancel', BTN_RED, win.destroy,
                    width=12, font=('Helvetica bold', 13)).place(x=340, y=btn_y + 48)

    # ── Run ───────────────────────────────────────────────────────────────

    def start(self):
        self.root.mainloop()
        if hasattr(self, 'cap'):
            self.cap.release()


if __name__ == "__main__":
    App().start()
