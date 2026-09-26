"""
Single-file 80% academic prototype:
Physics-Informed Neural Networks (PINNs) for Real-Time
Biomechanical Fatigue & Form Tracking.

Features:
- 33 gym + rehabilitation-support movements
- Webcam + recorded video
- MediaPipe pose estimation
- Joint angles, ROM, velocity, symmetry
- Rep counting / static hold
- Form score + feedback
- CSV dataset logging
- PyTorch PINN
- Data loss + physics loss
- Training, evaluation, saved model
- Real-time PINN fatigue prediction

IMPORTANT:
This is an academic prototype, not a medical diagnostic system.
Rehabilitation mode is for movement-monitoring support only.
"""

from pathlib import Path
import time
import sys
import re
import json
import os
import subprocess
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    GUI_AVAILABLE = True
except Exception:
    GUI_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DATA_PATH = DATA_DIR / "biomechanical_dataset.csv"
MODEL_PATH = MODEL_DIR / "pinn_fatigue.pt"
CONFIG_PATH = BASE_DIR / "exercise_config.json"
SESSIONS_DIR = RESULTS_DIR / "sessions"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"

for folder in (DATA_DIR, MODEL_DIR, RESULTS_DIR, SESSIONS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

LAST_GRAPH_PATHS = []
LAST_SUMMARY = None
LAST_SESSION_DIR = None


# ============================================================
# EXERCISE CONFIGURATION
# ============================================================

EXERCISE_CONFIGS = {
    # LOWER BODY
    "Squat": {"category": "Lower Body", "joint": "knee", "low": 115, "high": 155, "low_phase": "DOWN", "high_phase": "UP", "rep_on": "high", "symmetry_joint": "knee", "view": "front/side"},
    "Sumo Squat": {"category": "Lower Body", "joint": "knee", "low": 115, "high": 155, "low_phase": "DOWN", "high_phase": "UP", "rep_on": "high", "symmetry_joint": "knee", "view": "front"},
    "Lunge": {"category": "Lower Body", "joint": "knee", "low": 105, "high": 155, "low_phase": "DOWN", "high_phase": "UP", "rep_on": "high", "symmetry_joint": "knee", "view": "side"},
    "Reverse Lunge": {"category": "Lower Body", "joint": "knee", "low": 105, "high": 155, "low_phase": "DOWN", "high_phase": "UP", "rep_on": "high", "symmetry_joint": "knee", "view": "side"},
    "Bulgarian Split Squat": {"category": "Lower Body", "joint": "knee", "low": 105, "high": 155, "low_phase": "DOWN", "high_phase": "UP", "rep_on": "high", "symmetry_joint": "knee", "view": "side"},
    "Deadlift": {"category": "Lower Body", "joint": "hip", "low": 105, "high": 160, "low_phase": "HINGE", "high_phase": "STAND", "rep_on": "high", "symmetry_joint": "hip", "view": "side"},
    "Romanian Deadlift": {"category": "Lower Body", "joint": "hip", "low": 105, "high": 160, "low_phase": "HINGE", "high_phase": "STAND", "rep_on": "high", "symmetry_joint": "hip", "view": "side"},
    "Calf Raise": {"category": "Lower Body", "joint": "ankle", "low": 105, "high": 130, "low_phase": "LOW", "high_phase": "RAISED", "rep_on": "high", "symmetry_joint": "ankle", "view": "side"},
    "Step Up": {"category": "Lower Body", "joint": "knee", "low": 115, "high": 155, "low_phase": "STEP", "high_phase": "STAND", "rep_on": "high", "symmetry_joint": "knee", "view": "side"},

    # UPPER BODY
    "Bicep Curl": {"category": "Upper Body", "joint": "elbow", "low": 70, "high": 150, "low_phase": "CONTRACTED", "high_phase": "EXTENDED", "rep_on": "high", "symmetry_joint": "elbow", "view": "front/side"},
    "Hammer Curl": {"category": "Upper Body", "joint": "elbow", "low": 70, "high": 150, "low_phase": "CONTRACTED", "high_phase": "EXTENDED", "rep_on": "high", "symmetry_joint": "elbow", "view": "front/side"},
    "Tricep Extension": {"category": "Upper Body", "joint": "elbow", "low": 75, "high": 150, "low_phase": "BENT", "high_phase": "EXTENDED", "rep_on": "high", "symmetry_joint": "elbow", "view": "side"},
    "Shoulder Press": {"category": "Upper Body", "joint": "shoulder", "low": 70, "high": 145, "low_phase": "LOW", "high_phase": "PRESS", "rep_on": "high", "symmetry_joint": "shoulder", "view": "front"},
    "Lateral Raise": {"category": "Upper Body", "joint": "shoulder", "low": 35, "high": 80, "low_phase": "LOW", "high_phase": "RAISED", "rep_on": "high", "symmetry_joint": "shoulder", "view": "front"},
    "Front Raise": {"category": "Upper Body", "joint": "shoulder", "low": 35, "high": 80, "low_phase": "LOW", "high_phase": "RAISED", "rep_on": "high", "symmetry_joint": "shoulder", "view": "side/front"},
    "Shoulder Raise": {"category": "Upper Body", "joint": "shoulder", "low": 45, "high": 80, "low_phase": "LOW", "high_phase": "RAISED", "rep_on": "low", "symmetry_joint": "shoulder", "view": "front"},

    # CHEST / BACK / CORE
    "Push Up": {"category": "Chest/Core", "joint": "elbow", "low": 95, "high": 155, "low_phase": "DOWN", "high_phase": "UP", "rep_on": "high", "symmetry_joint": "elbow", "view": "side"},
    "Incline Push Up": {"category": "Chest/Core", "joint": "elbow", "low": 95, "high": 155, "low_phase": "DOWN", "high_phase": "UP", "rep_on": "high", "symmetry_joint": "elbow", "view": "side"},
    "Bent Over Row": {"category": "Back", "joint": "elbow", "low": 75, "high": 145, "low_phase": "PULL", "high_phase": "EXTEND", "rep_on": "high", "symmetry_joint": "elbow", "view": "side"},
    "Superman": {"category": "Back/Core", "joint": "hip", "low": 145, "high": 170, "low_phase": "LOW", "high_phase": "RAISED", "rep_on": "high", "symmetry_joint": "hip", "view": "side"},
    "Sit Up": {"category": "Core", "joint": "hip", "low": 95, "high": 145, "low_phase": "UP", "high_phase": "DOWN", "rep_on": "high", "symmetry_joint": "hip", "view": "side"},
    "Crunch": {"category": "Core", "joint": "hip", "low": 110, "high": 145, "low_phase": "UP", "high_phase": "DOWN", "rep_on": "high", "symmetry_joint": "hip", "view": "side"},
    "Plank": {"category": "Core", "joint": "hip", "static": True, "symmetry_joint": "hip", "view": "side"},
    "Side Plank": {"category": "Core", "joint": "hip", "static": True, "symmetry_joint": "hip", "view": "side"},
    "Mountain Climber": {"category": "Core", "joint": "knee", "low": 95, "high": 145, "low_phase": "DRIVE", "high_phase": "EXTEND", "rep_on": "high", "symmetry_joint": "knee", "view": "side"},

    # REHABILITATION SUPPORT
    "Knee Flexion": {"category": "Rehabilitation", "joint": "knee", "low": 100, "high": 155, "low_phase": "FLEXED", "high_phase": "EXTENDED", "rep_on": "high", "symmetry_joint": "knee", "view": "side"},
    "Knee Extension": {"category": "Rehabilitation", "joint": "knee", "low": 105, "high": 160, "low_phase": "BENT", "high_phase": "EXTENDED", "rep_on": "high", "symmetry_joint": "knee", "view": "side"},
    "Shoulder Flexion": {"category": "Rehabilitation", "joint": "shoulder", "low": 40, "high": 120, "low_phase": "LOW", "high_phase": "FLEXED", "rep_on": "high", "symmetry_joint": "shoulder", "view": "side"},
    "Shoulder Abduction": {"category": "Rehabilitation", "joint": "shoulder", "low": 40, "high": 90, "low_phase": "LOW", "high_phase": "ABDUCTED", "rep_on": "high", "symmetry_joint": "shoulder", "view": "front"},
    "Elbow Flexion": {"category": "Rehabilitation", "joint": "elbow", "low": 75, "high": 150, "low_phase": "FLEXED", "high_phase": "EXTENDED", "rep_on": "high", "symmetry_joint": "elbow", "view": "side"},
    "Hip Abduction": {"category": "Rehabilitation", "joint": "hip", "low": 145, "high": 165, "low_phase": "OUT", "high_phase": "NEUTRAL", "rep_on": "high", "symmetry_joint": "hip", "view": "front"},
    "Straight Leg Raise": {"category": "Rehabilitation", "joint": "hip", "low": 110, "high": 160, "low_phase": "RAISED", "high_phase": "LOW", "rep_on": "high", "symmetry_joint": "hip", "view": "side"},
    "Heel Raise": {"category": "Rehabilitation", "joint": "ankle", "low": 105, "high": 130, "low_phase": "LOW", "high_phase": "RAISED", "rep_on": "high", "symmetry_joint": "ankle", "view": "side"},
}


def save_default_exercise_config():
    """Create an editable JSON exercise configuration file."""
    if not CONFIG_PATH.exists():
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(EXERCISE_CONFIGS, f, indent=2)
        print("Created editable exercise config:", CONFIG_PATH)


def load_exercise_config():
    """
    Load thresholds from exercise_config.json when available.
    If the JSON is invalid, the built-in safe defaults remain active.
    """
    save_default_exercise_config()

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        if isinstance(loaded, dict) and loaded:
            valid = {}
            for name, cfg in loaded.items():
                if not isinstance(cfg, dict):
                    continue
                if "joint" not in cfg or "category" not in cfg or "view" not in cfg:
                    continue
                valid[name] = cfg

            if valid:
                EXERCISE_CONFIGS.clear()
                EXERCISE_CONFIGS.update(valid)

    except Exception as exc:
        print("Exercise config warning:", exc)


load_exercise_config()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# BIOMECHANICS
# ============================================================

def calculate_angle(a, b, c):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / denom
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def point(landmarks, landmark_id):
    lm = landmarks[landmark_id]
    return np.array([lm.x, lm.y], dtype=np.float32)


def extract_angles(landmarks):
    P = mp_pose.PoseLandmark

    ls, rs = point(landmarks, P.LEFT_SHOULDER.value), point(landmarks, P.RIGHT_SHOULDER.value)
    le, re = point(landmarks, P.LEFT_ELBOW.value), point(landmarks, P.RIGHT_ELBOW.value)
    lw, rw = point(landmarks, P.LEFT_WRIST.value), point(landmarks, P.RIGHT_WRIST.value)
    lh, rh = point(landmarks, P.LEFT_HIP.value), point(landmarks, P.RIGHT_HIP.value)
    lk, rk = point(landmarks, P.LEFT_KNEE.value), point(landmarks, P.RIGHT_KNEE.value)
    la, ra = point(landmarks, P.LEFT_ANKLE.value), point(landmarks, P.RIGHT_ANKLE.value)
    lf, rf = point(landmarks, P.LEFT_FOOT_INDEX.value), point(landmarks, P.RIGHT_FOOT_INDEX.value)

    return {
        "left_knee_angle": calculate_angle(lh, lk, la),
        "right_knee_angle": calculate_angle(rh, rk, ra),
        "left_hip_angle": calculate_angle(ls, lh, lk),
        "right_hip_angle": calculate_angle(rs, rh, rk),
        "left_elbow_angle": calculate_angle(ls, le, lw),
        "right_elbow_angle": calculate_angle(rs, re, rw),
        "left_shoulder_angle": calculate_angle(lh, ls, le),
        "right_shoulder_angle": calculate_angle(rh, rs, re),
        "left_ankle_angle": calculate_angle(lk, la, lf),
        "right_ankle_angle": calculate_angle(rk, ra, rf),
    }


JOINT_KEYS = {
    "knee": ("left_knee_angle", "right_knee_angle"),
    "hip": ("left_hip_angle", "right_hip_angle"),
    "elbow": ("left_elbow_angle", "right_elbow_angle"),
    "shoulder": ("left_shoulder_angle", "right_shoulder_angle"),
    "ankle": ("left_ankle_angle", "right_ankle_angle"),
}


def joint_average(angles, joint):
    left_key, right_key = JOINT_KEYS[joint]
    return (float(angles[left_key]) + float(angles[right_key])) / 2.0


def joint_symmetry(angles, joint):
    left_key, right_key = JOINT_KEYS[joint]
    return abs(float(angles[left_key]) - float(angles[right_key]))


def normalized_effort(velocity, rom, symmetry, reps):
    v = min(abs(float(velocity)) / 180.0, 1.0)
    r = min(abs(float(rom)) / 120.0, 1.0)
    s = min(abs(float(symmetry)) / 30.0, 1.0)
    rp = min(float(reps) / 20.0, 1.0)

    effort = 0.45 * v + 0.30 * r + 0.15 * (1.0 - s) + 0.10 * rp
    return float(np.clip(effort, 0.0, 1.0))



def form_score(symmetry, velocity, rom):
    """Generic base form score used by the exercise-specific evaluator."""
    score = 100.0
    score -= min(float(symmetry) * 1.25, 30.0)

    if velocity > 300:
        score -= 10.0

    return float(np.clip(score, 0.0, 100.0))


def form_status(score, symmetry):
    if symmetry > 20:
        return "ASYMMETRY"
    if score >= 85:
        return "GOOD FORM"
    if score >= 65:
        return "ACCEPTABLE"
    return "CHECK FORM"


def exercise_specific_form(
    exercise,
    angles,
    rom,
    velocity,
    symmetry,
    reps,
):
    """
    Exercise-specific academic form rules.

    These rules improve the prototype beyond a single generic score.
    They are not clinical diagnostic rules and should be calibrated
    with real exercise recordings before final validation.
    """
    cfg = EXERCISE_CONFIGS[exercise]
    score = form_score(symmetry, velocity, rom)
    feedback = []

    target_rom = 0.0
    if not cfg.get("static", False):
        target_rom = abs(float(cfg.get("high", 0)) - float(cfg.get("low", 0)))

    # General quality rules.
    if symmetry > 15:
        score -= 10
        feedback.append("Improve left-right symmetry")

    if velocity > 300:
        feedback.append("Use more controlled movement")

    if reps > 0 and target_rom > 0 and rom < target_rom * 0.65:
        score -= 12
        feedback.append("Increase movement range")

    knee = joint_average(angles, "knee")
    hip = joint_average(angles, "hip")
    elbow = joint_average(angles, "elbow")
    shoulder = joint_average(angles, "shoulder")

    squat_family = {
        "Squat", "Sumo Squat", "Lunge", "Reverse Lunge",
        "Bulgarian Split Squat", "Step Up"
    }
    curl_family = {"Bicep Curl", "Hammer Curl", "Elbow Flexion"}
    shoulder_raise_family = {
        "Lateral Raise", "Front Raise", "Shoulder Raise",
        "Shoulder Flexion", "Shoulder Abduction"
    }
    push_family = {"Push Up", "Incline Push Up"}
    hinge_family = {"Deadlift", "Romanian Deadlift"}

    if exercise in squat_family:
        if symmetry > 12:
            score -= 8
        if reps > 0 and rom < max(target_rom * 0.70, 25):
            score -= 8
            feedback.append("Check squat/lunge depth")

    elif exercise in curl_family:
        # Large shoulder movement during a curl often means the upper arm is swinging.
        if shoulder > 65:
            score -= 10
            feedback.append("Keep upper arm more stable")

    elif exercise == "Tricep Extension":
        if shoulder < 35:
            score -= 6
            feedback.append("Check upper-arm position")

    elif exercise in shoulder_raise_family:
        if elbow < 115:
            score -= 8
            feedback.append("Avoid excessive elbow bending")

    elif exercise == "Shoulder Press":
        if symmetry > 12:
            score -= 8
            feedback.append("Press both arms evenly")

    elif exercise in push_family:
        # A very small hip angle can indicate trunk sag/pike in a side camera view.
        if hip < 145:
            score -= 15
            feedback.append("Keep trunk more aligned")

    elif exercise in hinge_family:
        if knee < 105:
            score -= 8
            feedback.append("Check hip-hinge pattern")

    elif exercise in {"Plank", "Side Plank"}:
        if not 150 <= hip <= 180:
            score -= 20
            feedback.append("Keep trunk and hip aligned")

    elif exercise in {"Knee Flexion", "Knee Extension"}:
        if velocity > 220:
            score -= 8
            feedback.append("Use controlled rehabilitation movement")

    # Conservative rehabilitation rule: emphasize symmetry and control.
    if cfg.get("category") == "Rehabilitation":
        if symmetry > 12:
            score -= 8
        if velocity > 240:
            score -= 6

    score = float(np.clip(score, 0.0, 100.0))
    status = form_status(score, symmetry)

    if not feedback:
        feedback.append("Movement quality looks stable")

    return score, status, " | ".join(feedback[:2])


# ============================================================
# EXERCISE ENGINE
# ============================================================

class ExerciseAnalyzer:
    def __init__(self, exercise):
        self.exercise = exercise
        self.config = EXERCISE_CONFIGS[exercise]
        self.reps = 0
        self.phase = "START"
        self.static_hold_seconds = 0.0

    def reset(self):
        self.reps = 0
        self.phase = "START"
        self.static_hold_seconds = 0.0

    def update(self, angles, dt=0.0):
        cfg = self.config
        active_angle = joint_average(angles, cfg["joint"])

        if cfg.get("static", False):
            self.phase = "HOLD"
            self.static_hold_seconds += max(float(dt), 0.0)
            return self.reps, self.phase, active_angle

        previous_phase = self.phase

        if active_angle <= cfg["low"]:
            self.phase = cfg["low_phase"]
        elif active_angle >= cfg["high"]:
            self.phase = cfg["high_phase"]

        if cfg.get("rep_on", "high") == "high":
            if previous_phase == cfg["low_phase"] and self.phase == cfg["high_phase"]:
                self.reps += 1
        else:
            if previous_phase == cfg["high_phase"] and self.phase == cfg["low_phase"]:
                self.reps += 1

        return self.reps, self.phase, active_angle


# ============================================================
# DATA LOGGER
# ============================================================


class DataLogger:
    def __init__(self, session_id, session_dir):
        self.session_id = session_id
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.pending_rows = []
        self.session_rows = []

    def add(self, row):
        row = dict(row)
        row["session_id"] = self.session_id
        self.pending_rows.append(row)
        self.session_rows.append(row)

    def save(self):
        """Append only newly collected rows to the master training dataset."""
        if not self.pending_rows:
            return None

        new_df = pd.DataFrame(self.pending_rows)

        if DATA_PATH.exists():
            old_df = pd.read_csv(DATA_PATH)
            df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df = new_df

        df.to_csv(DATA_PATH, index=False)
        self.pending_rows.clear()
        print("Saved master dataset:", DATA_PATH)
        return DATA_PATH

    def save_session_csv(self, filename="session.csv"):
        if not self.session_rows:
            return None

        path = self.session_dir / filename
        pd.DataFrame(self.session_rows).to_csv(path, index=False)
        print("Saved session dataset:", path)
        return path


# ============================================================
# INPUT SOURCE
# ============================================================


def choose_input_source(source_type=None, video_path=None):
    """
    Open webcam/video. Optional arguments let the Tkinter dashboard
    call the same backend without terminal input.
    """
    if source_type is None:
        print("\nChoose input source:")
        print("1 - Live Webcam")
        print("2 - Recorded / Uploaded Video")
        choice = input("Choose input [1/2]: ").strip()
        source_type = "video" if choice == "2" else "webcam"

    source_type = str(source_type).lower()

    if source_type == "video":
        while True:
            if video_path:
                raw = str(video_path)
            else:
                raw = input(
                    "Enter full video path "
                    "(or drag the video file into Terminal): "
                )

            raw = raw.strip().strip('"').strip("'")
            path = Path(raw).expanduser()

            if not path.exists():
                if video_path:
                    raise FileNotFoundError(f"Video not found: {path}")
                print("Video not found:", path)
                continue

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()
                if video_path:
                    raise RuntimeError(f"Could not open video: {path}")
                print("Could not open video.")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 30.0

            return cap, "video", path.name, float(fps)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Webcam could not be opened. On macOS enable Camera permission "
            "for Visual Studio Code or Terminal."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    return cap, "webcam", "Live Webcam", float(fps)


def source_timestamp(cap, source_type, webcam_start):
    if source_type == "video":
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ms is not None and ms >= 0:
            return float(ms) / 1000.0

    return time.time() - webcam_start


def display_delay_ms(source_type, fps):
    if source_type == "video" and fps > 0:
        return max(1, int(round(1000.0 / fps)))
    return 1


# ============================================================
# PINN MODEL
# ============================================================

class FatiguePINN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.log_alpha = nn.Parameter(torch.tensor(-2.0))
        self.log_beta = nn.Parameter(torch.tensor(-2.0))

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def forward(self, x):
        return self.network(x)

    def physics_residual(self, x, time_index, effort_index):
        fatigue = self.forward(x)

        grad = torch.autograd.grad(
            outputs=fatigue,
            inputs=x,
            grad_outputs=torch.ones_like(fatigue),
            create_graph=True,
            retain_graph=True,
        )[0]

        dF_dt = grad[:, time_index:time_index + 1]
        effort = x[:, effort_index:effort_index + 1]

        # Physics equation:
        # dF/dt = alpha * effort - beta * fatigue
        return dF_dt - (self.alpha * effort - self.beta * fatigue)


FEATURES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "knee_rom",
    "hip_rom",
    "movement_velocity",
    "symmetry_difference",
    "effort",
    "repetitions",
    "time",
]

STANDARDIZED_FEATURES = [
    f for f in FEATURES
    if f not in ["effort", "time"]
]


# ============================================================
# MENUS
# ============================================================

def choose_mode():
    print("\n==============================")
    print("PINN BIOMECHANICAL TRACKER")
    print("==============================")
    print("1 - GYM MODE")
    print("2 - REHABILITATION MODE")

    return (
        "REHABILITATION"
        if input("Choose mode [1/2]: ").strip() == "2"
        else "GYM"
    )


def names_for_mode(mode):
    if mode == "REHABILITATION":
        return [
            name for name, cfg in EXERCISE_CONFIGS.items()
            if cfg["category"] == "Rehabilitation"
        ]

    return [
        name for name, cfg in EXERCISE_CONFIGS.items()
        if cfg["category"] != "Rehabilitation"
    ]


def choose_exercise(mode=None):
    names = list(EXERCISE_CONFIGS.keys()) if mode is None else names_for_mode(mode)

    print("\nChoose exercise:")

    for i, name in enumerate(names, 1):
        cfg = EXERCISE_CONFIGS[name]
        print(f"{i:2d} - {name}  [view: {cfg['view']}]")

    raw = input("Exercise number: ").strip()

    try:
        idx = int(raw) - 1
        if 0 <= idx < len(names):
            return names[idx]
    except ValueError:
        pass

    return names[0]


def put(frame, text, y, scale=0.52):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )



# ============================================================
# VIDEO-SPECIFIC SESSION RESULTS / GRAPHS
# ============================================================


def safe_filename(text):
    text = Path(str(text)).stem
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return text.strip("_") or "session"


def create_session_id(exercise, tag="session"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{safe_filename(exercise)}_{tag}"


def build_session_summary(
    rows,
    exercise,
    source_name,
    mode,
    fatigue_column,
    session_dir,
):
    if not rows:
        return None

    df = pd.DataFrame(rows).copy()
    if df.empty:
        return None

    duration = 0.0
    if "time" in df.columns and len(df) > 1:
        duration = max(0.0, float(df["time"].iloc[-1]) - float(df["time"].iloc[0]))

    def avg(column):
        return float(pd.to_numeric(df[column], errors="coerce").dropna().mean()) if column in df else 0.0

    def maxv(column):
        return float(pd.to_numeric(df[column], errors="coerce").dropna().max()) if column in df else 0.0

    summary = {
        "exercise": exercise,
        "mode": mode,
        "source": source_name,
        "duration_seconds": round(duration, 2),
        "total_repetitions": int(maxv("repetitions")),
        "static_hold_seconds": round(maxv("static_hold_seconds"), 2),
        "average_rom_deg": round(avg("active_rom"), 2),
        "maximum_rom_deg": round(maxv("active_rom"), 2),
        "average_velocity_deg_s": round(avg("movement_velocity"), 2),
        "average_symmetry_difference_deg": round(avg("symmetry_difference"), 2),
        "average_form_score_percent": round(avg("form_score"), 2),
        "average_fatigue_percent": round(avg(fatigue_column), 2),
        "peak_fatigue_percent": round(maxv(fatigue_column), 2),
    }

    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    json_path = session_dir / "summary.json"
    txt_path = session_dir / "summary.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("BIOFORM AI - SESSION SUMMARY\n")
        f.write("=" * 42 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("\nSESSION SUMMARY")
    print("-" * 42)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Saved summary:", txt_path)

    return summary


def generate_video_session_graphs(
    rows,
    source_name,
    exercise,
    fatigue_column,
    fatigue_label,
    output_tag,
    session_dir,
):
    global LAST_GRAPH_PATHS

    if not rows:
        print("No session data available for graphs.")
        return []

    df = pd.DataFrame(rows).copy()

    required = [
        "time",
        "form_score",
        "active_rom",
        "symmetry_difference",
        fatigue_column,
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        print("Could not create all graphs. Missing columns:", missing)
        return []

    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        return []

    start_time = float(df["time"].iloc[0])
    df["session_time"] = df["time"].astype(float) - start_time

    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    session_csv = session_dir / f"{output_tag}_session.csv"
    df.to_csv(session_csv, index=False)

    graph_specs = [
        (fatigue_column, fatigue_label, "Fatigue (%)", "fatigue"),
        ("form_score", "Form Score", "Form Score (%)", "form"),
        ("active_rom", "Range of Motion", "ROM (degrees)", "rom"),
        ("symmetry_difference", "Bilateral Symmetry Difference", "Difference (degrees)", "symmetry"),
        ("movement_velocity", "Movement Velocity", "Velocity (deg/s)", "velocity"),
    ]

    saved_paths = []

    for column, title_name, ylabel, suffix in graph_specs:
        if column not in df.columns:
            continue

        plt.figure(figsize=(9, 5))
        plt.plot(df["session_time"], df[column])
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{title_name} vs Time - {exercise}")
        plt.tight_layout()

        output = session_dir / f"{output_tag}_{suffix}.png"
        plt.savefig(output, dpi=180)
        plt.close()
        saved_paths.append(output)

    LAST_GRAPH_PATHS = saved_paths

    print("\nSESSION GRAPHS GENERATED")
    print("-" * 42)
    for path in saved_paths:
        print("Saved:", path)

    return saved_paths


# ============================================================
# 1. BIOMECHANICAL TRACKING / DATA COLLECTION
# ============================================================


def run_tracking(
    mode=None,
    exercise=None,
    source_type=None,
    video_path=None,
):
    global LAST_SUMMARY, LAST_SESSION_DIR

    mode = mode or choose_mode()
    exercise = exercise or choose_exercise(mode)

    if exercise not in EXERCISE_CONFIGS:
        raise ValueError(f"Unknown exercise: {exercise}")

    cfg = EXERCISE_CONFIGS[exercise]

    cap, source_type, source_name, fps = choose_input_source(
        source_type=source_type,
        video_path=video_path,
    )

    session_id = create_session_id(exercise, "biomechanical")
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    LAST_SESSION_DIR = session_dir

    analyzer = ExerciseAnalyzer(exercise)
    logger = DataLogger(session_id, session_dir)

    webcam_start = time.time()
    previous_time = None
    previous_active_angle = None

    active_history = []
    knee_history = []
    hip_history = []

    fatigue_proxy = 0.0
    no_pose_frames = 0

    print("\nExercise:", exercise)
    print("Recommended camera view:", cfg["view"])
    print("Input:", source_name)
    print("Session:", session_id)

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while True:
                ok, frame = cap.read()

                if not ok:
                    if source_type == "video":
                        print("\nVideo analysis completed.")
                    break

                if source_type == "webcam":
                    frame = cv2.flip(frame, 1)

                results = pose.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                timestamp = source_timestamp(
                    cap,
                    source_type,
                    webcam_start,
                )

                dt = (
                    timestamp - previous_time
                    if previous_time is not None
                    else 0.0
                )

                if results.pose_landmarks:
                    no_pose_frames = 0

                    angles = extract_angles(
                        results.pose_landmarks.landmark
                    )

                    reps, phase, active_angle = analyzer.update(
                        angles,
                        dt=dt,
                    )

                    active_history.append(active_angle)
                    knee_history.append(joint_average(angles, "knee"))
                    hip_history.append(joint_average(angles, "hip"))

                    velocity = (
                        abs(active_angle - previous_active_angle) / dt
                        if previous_active_angle is not None and dt > 1e-4
                        else 0.0
                    )

                    previous_time = timestamp
                    previous_active_angle = active_angle

                    rom = max(active_history) - min(active_history)
                    knee_rom = max(knee_history) - min(knee_history)
                    hip_rom = max(hip_history) - min(hip_history)

                    symmetry = joint_symmetry(
                        angles,
                        cfg["symmetry_joint"],
                    )

                    effort = normalized_effort(
                        velocity,
                        rom,
                        symmetry,
                        reps,
                    )

                    fatigue_proxy = float(
                        np.clip(
                            fatigue_proxy * 0.985 + effort * 1.8,
                            0.0,
                            100.0,
                        )
                    )

                    score, status, feedback = exercise_specific_form(
                        exercise,
                        angles,
                        rom,
                        velocity,
                        symmetry,
                        reps,
                    )

                    logger.add({
                        "time": timestamp,
                        "input_source": source_type,
                        "source_name": source_name,
                        "mode": mode,
                        "exercise": exercise,
                        "category": cfg["category"],
                        **angles,
                        "active_joint": cfg["joint"],
                        "active_angle": active_angle,
                        "active_rom": rom,
                        "knee_rom": knee_rom,
                        "hip_rom": hip_rom,
                        "movement_velocity": velocity,
                        "symmetry_difference": symmetry,
                        "effort": effort,
                        "repetitions": reps,
                        "static_hold_seconds": analyzer.static_hold_seconds,
                        "form_score": score,
                        "fatigue_score": fatigue_proxy,
                        "form_status": status,
                        "feedback": feedback,
                    })

                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay,
                        (10, 10),
                        (690, 470),
                        (0, 0, 0),
                        -1,
                    )

                    frame = cv2.addWeighted(
                        overlay,
                        0.70,
                        frame,
                        0.30,
                        0,
                    )

                    put(frame, f"MODE: {mode}", 38)
                    put(frame, f"EXERCISE: {exercise}", 68)
                    put(frame, f"VIEW: {cfg['view']}", 98)
                    put(frame, f"INPUT: {source_name}", 128)
                    put(frame, f"REPS: {reps}  PHASE: {phase}", 158)

                    if cfg.get("static", False):
                        put(
                            frame,
                            f"HOLD: {analyzer.static_hold_seconds:.1f} sec",
                            188,
                        )
                    else:
                        put(
                            frame,
                            f"{cfg['joint'].upper()} ANGLE: {active_angle:.1f} deg",
                            188,
                        )

                    put(frame, f"ROM: {rom:.1f} deg", 218)
                    put(frame, f"VELOCITY: {velocity:.1f} deg/s", 248)
                    put(frame, f"SYMMETRY: {symmetry:.1f} deg", 278)
                    put(frame, f"FORM: {score:.0f}% - {status}", 308)
                    put(frame, f"FATIGUE PROXY: {fatigue_proxy:.1f}%", 338)
                    put(frame, f"FEEDBACK: {feedback[:55]}", 368, 0.42)

                    if mode == "REHABILITATION":
                        put(
                            frame,
                            "Rehab support only - not medical diagnosis",
                            398,
                            0.42,
                        )

                    put(frame, "Q/ESC Quit | S Save | R Reset", 438, 0.45)

                else:
                    no_pose_frames += 1
                    cv2.putText(
                        frame,
                        "NO POSE DETECTED - keep full body visible",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(
                    "BioForm AI - Biomechanical Tracking",
                    frame,
                )

                key = cv2.waitKey(
                    display_delay_ms(source_type, fps)
                ) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord("s"):
                    logger.save()
                    logger.save_session_csv()

                if key == ord("r"):
                    analyzer.reset()
                    fatigue_proxy = 0.0
                    previous_time = None
                    previous_active_angle = None
                    active_history.clear()
                    knee_history.clear()
                    hip_history.clear()
                    print("Session counters reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.save()
    logger.save_session_csv()

    graph_paths = generate_video_session_graphs(
        rows=logger.session_rows,
        source_name=source_name,
        exercise=exercise,
        fatigue_column="fatigue_score",
        fatigue_label="Fatigue Proxy",
        output_tag="biomechanical",
        session_dir=session_dir,
    )

    LAST_SUMMARY = build_session_summary(
        rows=logger.session_rows,
        exercise=exercise,
        source_name=source_name,
        mode=mode,
        fatigue_column="fatigue_score",
        session_dir=session_dir,
    )

    return {
        "summary": LAST_SUMMARY,
        "graphs": graph_paths,
        "session_dir": session_dir,
    }


# ============================================================
# SYNTHETIC DATA FALLBACK
# ============================================================

def make_demo_data(n=3000):
    rng = np.random.default_rng(42)

    t = np.linspace(0, 180, n)

    knee = 130 + 35 * np.sin(t / 2.8) + rng.normal(0, 3, n)
    knee_r = knee + rng.normal(0, 4, n)

    hip = 140 + 25 * np.sin(t / 2.8 + 0.2) + rng.normal(0, 3, n)
    hip_r = hip + rng.normal(0, 4, n)

    elbow = 110 + 45 * np.sin(t / 2.0) + rng.normal(0, 4, n)
    elbow_r = elbow + rng.normal(0, 4, n)

    shoulder = 70 + 25 * np.sin(t / 2.5) + rng.normal(0, 3, n)
    shoulder_r = shoulder + rng.normal(0, 3, n)

    ankle = 115 + 10 * np.sin(t / 2.8) + rng.normal(0, 2, n)
    ankle_r = ankle + rng.normal(0, 2, n)

    velocity = np.abs(np.gradient(knee, t)) * 20
    symmetry = np.abs(knee - knee_r)

    knee_s = pd.Series(knee)
    hip_s = pd.Series(hip)

    knee_rom = (
        knee_s.rolling(80, min_periods=1).max()
        - knee_s.rolling(80, min_periods=1).min()
    )

    hip_rom = (
        hip_s.rolling(80, min_periods=1).max()
        - hip_s.rolling(80, min_periods=1).min()
    )

    effort = np.clip(
        0.45 * np.minimum(velocity / 180, 1)
        + 0.30 * np.minimum(knee_rom.values / 120, 1)
        + 0.15 * (1 - np.minimum(symmetry / 30, 1)),
        0,
        1,
    )

    reps = np.floor(t / 8).astype(float)

    fatigue = np.zeros(n)

    for i in range(1, n):
        dt = t[i] - t[i - 1]

        fatigue[i] = np.clip(
            fatigue[i - 1]
            + (
                0.12 * effort[i]
                - 0.008 * (fatigue[i - 1] / 100.0)
            )
            * dt
            * 100,
            0,
            100,
        )

    return pd.DataFrame({
        "left_knee_angle": knee,
        "right_knee_angle": knee_r,
        "left_hip_angle": hip,
        "right_hip_angle": hip_r,
        "left_elbow_angle": elbow,
        "right_elbow_angle": elbow_r,
        "left_shoulder_angle": shoulder,
        "right_shoulder_angle": shoulder_r,
        "left_ankle_angle": ankle,
        "right_ankle_angle": ankle_r,
        "knee_rom": knee_rom.values,
        "hip_rom": hip_rom.values,
        "movement_velocity": velocity,
        "symmetry_difference": symmetry,
        "effort": effort,
        "repetitions": reps,
        "time": t,
        "fatigue_score": fatigue,
    })


# ============================================================
# 2. PINN TRAINING
# ============================================================


def load_training_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print("Using collected dataset:", DATA_PATH)
    else:
        print("No collected dataset found.")
        print("Using synthetic demonstration dataset.")
        df = make_demo_data()

    required = FEATURES + ["fatigue_score"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print("Existing CSV is missing columns:", missing)
        print("Using synthetic demonstration dataset.")
        df = make_demo_data()

    # Session identifier enables session-level splitting.
    if "session_id" not in df.columns:
        if "source_name" in df.columns:
            source = df["source_name"].fillna("legacy").astype(str)
        else:
            source = pd.Series(["legacy"] * len(df))

        if "exercise" in df.columns:
            exercise = df["exercise"].fillna("exercise").astype(str)
        else:
            exercise = pd.Series(["exercise"] * len(df))

        df["session_id"] = source + "_" + exercise

    df = df[required + ["session_id"]].dropna().reset_index(drop=True)
    df["fatigue_score"] = np.clip(df["fatigue_score"], 0, 100) / 100.0

    return df


def _transform_frame(df, scaler, time_min, time_max):
    out = df.copy()

    out[STANDARDIZED_FEATURES] = scaler.transform(
        df[STANDARDIZED_FEATURES]
    )

    span = max(float(time_max) - float(time_min), 1e-8)
    out["time"] = (df["time"] - time_min) / span
    out["effort"] = np.clip(df["effort"], 0, 1)

    return out


def train_pinn():
    global LAST_GRAPH_PATHS

    torch.manual_seed(42)
    np.random.seed(42)

    df = load_training_data()

    # Prefer a session/video-level split so frames from the same session
    # are not mixed between train and test.
    unique_sessions = df["session_id"].nunique()

    if unique_sessions >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=42,
        )
        train_idx, test_idx = next(
            splitter.split(df, groups=df["session_id"])
        )
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        split_method = "session-level GroupShuffleSplit"
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=42,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        split_method = "random split fallback (only one session available)"

    print("Split method:", split_method)
    print("Training sessions:", train_df["session_id"].nunique())
    print("Testing sessions:", test_df["session_id"].nunique())

    scaler = StandardScaler()
    scaler.fit(train_df[STANDARDIZED_FEATURES])

    time_min = float(train_df["time"].min())
    time_max = float(train_df["time"].max())

    train_scaled = _transform_frame(
        train_df,
        scaler,
        time_min,
        time_max,
    )
    test_scaled = _transform_frame(
        test_df,
        scaler,
        time_min,
        time_max,
    )

    Xtr_np = train_scaled[FEATURES].values.astype(np.float32)
    ytr_np = train_scaled["fatigue_score"].values.astype(np.float32).reshape(-1, 1)

    Xte_np = test_scaled[FEATURES].values.astype(np.float32)
    yte_np = test_scaled["fatigue_score"].values.astype(np.float32).reshape(-1, 1)

    device = (
        torch.device("mps")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        else torch.device("cpu")
    )

    print("\nTraining device:", device)

    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr_np, dtype=torch.float32, device=device)
    Xte = torch.tensor(Xte_np, dtype=torch.float32, device=device)
    yte = torch.tensor(yte_np, dtype=torch.float32, device=device)

    model = FatiguePINN(len(FEATURES)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    time_idx = FEATURES.index("time")
    effort_idx = FEATURES.index("effort")

    total_losses = []
    data_losses = []
    physics_losses = []

    print("\nTraining PINN for 1000 epochs...")

    for epoch in range(1, 1001):
        model.train()

        x = Xtr.clone().detach().requires_grad_(True)
        prediction = model(x)

        data_loss = torch.mean((prediction - ytr) ** 2)

        residual = model.physics_residual(
            x,
            time_idx,
            effort_idx,
        )

        physics_loss = torch.mean(residual ** 2)
        loss = data_loss + 0.20 * physics_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_losses.append(loss.item())
        data_losses.append(data_loss.item())
        physics_losses.append(physics_loss.item())

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Total={loss.item():.6f} | "
                f"Data={data_loss.item():.6f} | "
                f"Physics={physics_loss.item():.6f} | "
                f"alpha={model.alpha.item():.4f} | "
                f"beta={model.beta.item():.4f}"
            )

    model.eval()

    with torch.no_grad():
        prediction = model(Xte)

    actual_np = yte.cpu().numpy().ravel()
    predicted_np = prediction.cpu().numpy().ravel()

    mse = float(np.mean((predicted_np - actual_np) ** 2))
    mae = float(np.mean(np.abs(predicted_np - actual_np)))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(actual_np, predicted_np))

    print("\nMODEL EVALUATION")
    print("MSE:", round(mse, 6))
    print("RMSE:", round(rmse, 6))
    print("MAE:", round(mae, 6))
    print("MAE percentage points:", round(mae * 100, 2))
    print("R2 Score:", round(r2, 4))

    checkpoint = {
        "model_state": model.state_dict(),
        "features": FEATURES,
        "standardized_features": STANDARDIZED_FEATURES,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "time_min": time_min,
        "time_max": time_max,
        "split_method": split_method,
    }

    # Latest model for inference.
    torch.save(checkpoint, MODEL_PATH)

    # Versioned model for reproducibility/comparison.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_model_path = MODEL_DIR / f"pinn_fatigue_{stamp}.pt"
    torch.save(checkpoint, versioned_model_path)

    evaluation_df = pd.DataFrame({
        "actual_fatigue_percent": actual_np * 100,
        "predicted_fatigue_percent": predicted_np * 100,
        "test_session_id": test_df["session_id"].astype(str).values,
    })

    evaluation_path = RESULTS_DIR / "evaluation.csv"
    evaluation_df.to_csv(evaluation_path, index=False)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mae_percentage_points": mae * 100,
        "r2": r2,
        "split_method": split_method,
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "training_sessions": int(train_df["session_id"].nunique()),
        "testing_sessions": int(test_df["session_id"].nunique()),
        "latest_model": str(MODEL_PATH),
        "versioned_model": str(versioned_model_path),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(9, 5))
    plt.plot(total_losses, label="Total Loss")
    plt.plot(data_losses, label="Data Loss")
    plt.plot(physics_losses, label="Physics Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = RESULTS_DIR / "training_loss.png"
    plt.savefig(loss_path, dpi=180)
    plt.close()

    LAST_GRAPH_PATHS = [loss_path]

    print("\nSaved latest model:", MODEL_PATH)
    print("Saved versioned model:", versioned_model_path)
    print("Saved metrics:", METRICS_PATH)
    print("Saved evaluation:", evaluation_path)
    print("Saved graph:", loss_path)

    return metrics


# ============================================================
# MODEL LOADER
# ============================================================

def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run option 2 (Train PINN) first."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    features = checkpoint[
        "features"
    ]

    standardized_features = checkpoint[
        "standardized_features"
    ]

    model = FatiguePINN(
        len(features)
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    return {
        "model":
            model,
        "features":
            features,
        "standardized_features":
            standardized_features,
        "mean":
            np.asarray(
                checkpoint[
                    "scaler_mean"
                ]
            ),
        "scale":
            np.asarray(
                checkpoint[
                    "scaler_scale"
                ]
            ),
        "time_min":
            float(
                checkpoint[
                    "time_min"
                ]
            ),
        "time_max":
            float(
                checkpoint[
                    "time_max"
                ]
            ),
    }


# ============================================================
# 3. REAL-TIME TRAINED PINN PREDICTION
# ============================================================


def run_realtime_pinn(
    exercise=None,
    source_type=None,
    video_path=None,
    mode="GYM",
):
    global LAST_SUMMARY, LAST_SESSION_DIR

    bundle = load_model_bundle()

    exercise = exercise or choose_exercise(mode=None)

    if exercise not in EXERCISE_CONFIGS:
        raise ValueError(f"Unknown exercise: {exercise}")

    cfg = EXERCISE_CONFIGS[exercise]

    cap, source_type, source_name, fps = choose_input_source(
        source_type=source_type,
        video_path=video_path,
    )

    session_id = create_session_id(exercise, "pinn")
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    LAST_SESSION_DIR = session_dir

    analyzer = ExerciseAnalyzer(exercise)

    webcam_start = time.time()
    previous_time = None
    previous_active_angle = None

    active_history = []
    knee_history = []
    hip_history = []

    realtime_rows = []

    print("\nTrained PINN loaded.")
    print("Exercise:", exercise)
    print("Recommended view:", cfg["view"])
    print("Input:", source_name)
    print("PINN fatigue is an academic estimate, not a medical diagnosis.")

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while True:
                ok, frame = cap.read()

                if not ok:
                    if source_type == "video":
                        print("\nVideo PINN analysis completed.")
                    break

                if source_type == "webcam":
                    frame = cv2.flip(frame, 1)

                results = pose.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                timestamp = source_timestamp(
                    cap,
                    source_type,
                    webcam_start,
                )

                dt = (
                    timestamp - previous_time
                    if previous_time is not None
                    else 0.0
                )

                if results.pose_landmarks:
                    angles = extract_angles(
                        results.pose_landmarks.landmark
                    )

                    reps, phase, active_angle = analyzer.update(
                        angles,
                        dt=dt,
                    )

                    active_history.append(active_angle)
                    knee_history.append(joint_average(angles, "knee"))
                    hip_history.append(joint_average(angles, "hip"))

                    velocity = (
                        abs(active_angle - previous_active_angle) / dt
                        if previous_active_angle is not None and dt > 1e-4
                        else 0.0
                    )

                    previous_time = timestamp
                    previous_active_angle = active_angle

                    rom = max(active_history) - min(active_history)
                    knee_rom = max(knee_history) - min(knee_history)
                    hip_rom = max(hip_history) - min(hip_history)

                    symmetry = joint_symmetry(
                        angles,
                        cfg["symmetry_joint"],
                    )

                    effort = normalized_effort(
                        velocity,
                        rom,
                        symmetry,
                        reps,
                    )

                    score, status, feedback = exercise_specific_form(
                        exercise,
                        angles,
                        rom,
                        velocity,
                        symmetry,
                        reps,
                    )

                    normalized_time = np.clip(
                        (timestamp - bundle["time_min"])
                        / max(
                            bundle["time_max"] - bundle["time_min"],
                            1e-8,
                        ),
                        0,
                        1,
                    )

                    raw = {
                        "left_knee_angle": angles["left_knee_angle"],
                        "right_knee_angle": angles["right_knee_angle"],
                        "left_hip_angle": angles["left_hip_angle"],
                        "right_hip_angle": angles["right_hip_angle"],
                        "left_elbow_angle": angles["left_elbow_angle"],
                        "right_elbow_angle": angles["right_elbow_angle"],
                        "left_shoulder_angle": angles["left_shoulder_angle"],
                        "right_shoulder_angle": angles["right_shoulder_angle"],
                        "left_ankle_angle": angles["left_ankle_angle"],
                        "right_ankle_angle": angles["right_ankle_angle"],
                        "knee_rom": knee_rom,
                        "hip_rom": hip_rom,
                        "movement_velocity": velocity,
                        "symmetry_difference": symmetry,
                        "effort": effort,
                        "repetitions": reps,
                        "time": normalized_time,
                    }

                    x = np.array(
                        [[raw[f] for f in bundle["features"]]],
                        dtype=np.float32,
                    )

                    for j, feature in enumerate(bundle["features"]):
                        if feature in bundle["standardized_features"]:
                            idx = bundle["standardized_features"].index(feature)
                            x[0, j] = (
                                x[0, j] - bundle["mean"][idx]
                            ) / max(bundle["scale"][idx], 1e-8)

                    with torch.no_grad():
                        fatigue = (
                            bundle["model"](
                                torch.tensor(
                                    x,
                                    dtype=torch.float32,
                                )
                            ).item()
                            * 100
                        )

                    realtime_rows.append({
                        "session_id": session_id,
                        "time": timestamp,
                        "mode": mode,
                        "exercise": exercise,
                        "source_name": source_name,
                        "repetitions": reps,
                        "phase": phase,
                        "active_angle": active_angle,
                        "active_rom": rom,
                        "knee_rom": knee_rom,
                        "hip_rom": hip_rom,
                        "movement_velocity": velocity,
                        "symmetry_difference": symmetry,
                        "form_score": score,
                        "form_status": status,
                        "feedback": feedback,
                        "effort": effort,
                        "static_hold_seconds": analyzer.static_hold_seconds,
                        "pinn_fatigue": fatigue,
                    })

                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                    cv2.rectangle(
                        frame,
                        (10, 10),
                        (690, 400),
                        (0, 0, 0),
                        -1,
                    )

                    lines = [
                        f"EXERCISE: {exercise}",
                        f"VIEW: {cfg['view']}",
                        f"REPS: {reps}  PHASE: {phase}",
                        f"{cfg['joint'].upper()} ANGLE: {active_angle:.1f} deg",
                        f"ROM: {rom:.1f} deg",
                        f"SYMMETRY: {symmetry:.1f} deg",
                        f"FORM: {score:.0f}% - {status}",
                        f"PINN FATIGUE: {fatigue:.1f}%",
                        f"FEEDBACK: {feedback[:48]}",
                    ]

                    for i, line in enumerate(lines):
                        cv2.putText(
                            frame,
                            line,
                            (25, 42 + i * 38),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                else:
                    cv2.putText(
                        frame,
                        "NO POSE DETECTED - keep full body visible",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(
                    "BioForm AI - PINN Fatigue Prediction",
                    frame,
                )

                key = cv2.waitKey(
                    display_delay_ms(source_type, fps)
                ) & 0xFF

                if key in (ord("q"), 27):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if realtime_rows:
        pd.DataFrame(realtime_rows).to_csv(
            session_dir / "pinn_session.csv",
            index=False,
        )

    graph_paths = generate_video_session_graphs(
        rows=realtime_rows,
        source_name=source_name,
        exercise=exercise,
        fatigue_column="pinn_fatigue",
        fatigue_label="PINN Fatigue",
        output_tag="pinn",
        session_dir=session_dir,
    )

    LAST_SUMMARY = build_session_summary(
        rows=realtime_rows,
        exercise=exercise,
        source_name=source_name,
        mode=mode,
        fatigue_column="pinn_fatigue",
        session_dir=session_dir,
    )

    return {
        "summary": LAST_SUMMARY,
        "graphs": graph_paths,
        "session_dir": session_dir,
    }


# ============================================================
# 4. EVALUATION GRAPH
# ============================================================


def evaluate_model():
    global LAST_GRAPH_PATHS

    path = RESULTS_DIR / "evaluation.csv"

    if not path.exists():
        print(
            "Evaluation file not found. "
            "Run PINN training first."
        )
        return []

    df = pd.read_csv(path)

    if df.empty:
        print("Evaluation file is empty.")
        return []

    n = min(500, len(df))

    actual = df["actual_fatigue_percent"].values[:n]
    predicted = df["predicted_fatigue_percent"].values[:n]

    line_path = RESULTS_DIR / "fatigue_prediction.png"
    scatter_path = RESULTS_DIR / "fatigue_actual_vs_predicted_scatter.png"

    plt.figure(figsize=(9, 5))
    plt.plot(actual, label="Actual / Dataset")
    plt.plot(predicted, label="PINN Prediction")
    plt.xlabel("Test Sample")
    plt.ylabel("Fatigue (%)")
    plt.title("PINN Fatigue Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(line_path, dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.6)
    low = float(min(np.min(actual), np.min(predicted)))
    high = float(max(np.max(actual), np.max(predicted)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Fatigue (%)")
    plt.ylabel("Predicted Fatigue (%)")
    plt.title("Actual vs Predicted Fatigue")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=180)
    plt.close()

    LAST_GRAPH_PATHS = [line_path, scatter_path]

    print("Saved:", line_path)
    print("Saved:", scatter_path)

    return LAST_GRAPH_PATHS


# ============================================================
# MAIN MASTER MENU
# ============================================================


def project_status_text():
    metrics = None
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            metrics = None

    lines = [
        f"Supported movements: {len(EXERCISE_CONFIGS)}",
        f"Dataset: {'FOUND' if DATA_PATH.exists() else 'NOT YET CREATED'}",
        f"Trained model: {'FOUND' if MODEL_PATH.exists() else 'NOT YET CREATED'}",
        f"Exercise config: {CONFIG_PATH}",
        f"Session results folder: {SESSIONS_DIR}",
    ]

    if metrics:
        lines.extend([
            f"MAE: {metrics.get('mae', 0):.6f}",
            f"MAE percentage points: {metrics.get('mae_percentage_points', 0):.2f}",
            f"RMSE: {metrics.get('rmse', 0):.6f}",
            f"R2: {metrics.get('r2', 0):.4f}",
            f"Split: {metrics.get('split_method', '')}",
        ])

    return "\n".join(lines)


def open_path(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif os.name == "nt":
        os.startfile(str(path))
    else:
        subprocess.Popen(["xdg-open", str(path)])


def show_graph_viewer(graph_paths=None):
    if not GUI_AVAILABLE:
        print("Tkinter GUI is not available.")
        return

    paths = [Path(p) for p in (graph_paths or LAST_GRAPH_PATHS) if Path(p).exists()]

    if not paths:
        messagebox.showinfo(
            "Graphs",
            "No graphs are available yet. Run an analysis or evaluation first.",
        )
        return

    win = tk.Toplevel()
    win.title("BioForm AI - Graph Viewer")
    win.geometry("1000x720")

    top = ttk.Frame(win)
    top.pack(fill="x", padx=10, pady=10)

    graph_var = tk.StringVar(value=str(paths[0]))
    combo = ttk.Combobox(
        top,
        textvariable=graph_var,
        values=[str(p) for p in paths],
        state="readonly",
        width=90,
    )
    combo.pack(side="left", fill="x", expand=True)

    figure = Figure(figsize=(9, 6), dpi=100)
    axis = figure.add_subplot(111)
    axis.axis("off")

    canvas = FigureCanvasTkAgg(figure, master=win)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def render_selected(*_):
        axis.clear()
        axis.axis("off")
        image = plt.imread(graph_var.get())
        axis.imshow(image)
        axis.set_title(Path(graph_var.get()).name)
        canvas.draw()

    combo.bind("<<ComboboxSelected>>", render_selected)
    render_selected()


def _summary_message(summary):
    if not summary:
        return "Session completed."

    return (
        f"Exercise: {summary.get('exercise')}\n"
        f"Repetitions: {summary.get('total_repetitions')}\n"
        f"Average ROM: {summary.get('average_rom_deg')}°\n"
        f"Average Form Score: {summary.get('average_form_score_percent')}%\n"
        f"Average Symmetry Difference: "
        f"{summary.get('average_symmetry_difference_deg')}°\n"
        f"Average Fatigue: {summary.get('average_fatigue_percent')}%\n"
        f"Peak Fatigue: {summary.get('peak_fatigue_percent')}%"
    )


def launch_dashboard():
    if not GUI_AVAILABLE:
        print("Tkinter is not available. Starting terminal menu.")
        master_menu()
        return

    root = tk.Tk()
    root.title("BioForm AI")
    root.geometry("820x720")

    title = ttk.Label(
        root,
        text="BioForm AI\nPhysics-Informed Biomechanical Fatigue & Form Tracking",
        font=("Arial", 20, "bold"),
        justify="center",
    )
    title.pack(pady=18)

    form = ttk.LabelFrame(root, text="Analysis Settings")
    form.pack(fill="x", padx=25, pady=10)

    mode_var = tk.StringVar(value="GYM")
    exercise_var = tk.StringVar()
    source_var = tk.StringVar(value="webcam")
    video_var = tk.StringVar()

    ttk.Label(form, text="Mode").grid(row=0, column=0, sticky="w", padx=10, pady=8)
    mode_combo = ttk.Combobox(
        form,
        textvariable=mode_var,
        values=["GYM", "REHABILITATION"],
        state="readonly",
        width=24,
    )
    mode_combo.grid(row=0, column=1, sticky="w", padx=10, pady=8)

    ttk.Label(form, text="Exercise").grid(row=1, column=0, sticky="w", padx=10, pady=8)
    exercise_combo = ttk.Combobox(
        form,
        textvariable=exercise_var,
        state="readonly",
        width=32,
    )
    exercise_combo.grid(row=1, column=1, sticky="w", padx=10, pady=8)

    def refresh_exercises(*_):
        names = names_for_mode(mode_var.get())
        exercise_combo["values"] = names
        if names:
            exercise_var.set(names[0])

    mode_combo.bind("<<ComboboxSelected>>", refresh_exercises)
    refresh_exercises()

    ttk.Label(form, text="Input").grid(row=2, column=0, sticky="w", padx=10, pady=8)

    source_frame = ttk.Frame(form)
    source_frame.grid(row=2, column=1, sticky="w", padx=10, pady=8)

    ttk.Radiobutton(
        source_frame,
        text="Live Webcam",
        variable=source_var,
        value="webcam",
    ).pack(side="left", padx=5)

    ttk.Radiobutton(
        source_frame,
        text="Video",
        variable=source_var,
        value="video",
    ).pack(side="left", padx=5)

    ttk.Label(form, text="Video file").grid(row=3, column=0, sticky="w", padx=10, pady=8)

    video_frame = ttk.Frame(form)
    video_frame.grid(row=3, column=1, sticky="ew", padx=10, pady=8)

    video_entry = ttk.Entry(
        video_frame,
        textvariable=video_var,
        width=48,
    )
    video_entry.pack(side="left", fill="x", expand=True)

    def browse_video():
        path = filedialog.askopenfilename(
            title="Choose exercise video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if path:
            video_var.set(path)
            source_var.set("video")

    ttk.Button(
        video_frame,
        text="Browse",
        command=browse_video,
    ).pack(side="left", padx=6)

    status_var = tk.StringVar(value="Ready")
    status = ttk.Label(
        root,
        textvariable=status_var,
        anchor="center",
    )
    status.pack(fill="x", padx=25, pady=8)

    buttons = ttk.LabelFrame(root, text="Project Controls")
    buttons.pack(fill="both", expand=True, padx=25, pady=10)

    def chosen_video():
        if source_var.get() == "video":
            if not video_var.get().strip():
                raise ValueError("Choose a video file first.")
            return video_var.get().strip()
        return None

    def run_with_dashboard(action):
        try:
            status_var.set("Running...")
            root.update_idletasks()
            root.withdraw()

            result = action()

            root.deiconify()
            root.lift()
            status_var.set("Completed")

            if isinstance(result, dict) and result.get("summary"):
                messagebox.showinfo(
                    "Session Summary",
                    _summary_message(result["summary"]),
                )

        except Exception as exc:
            root.deiconify()
            root.lift()
            status_var.set("Error")
            messagebox.showerror("BioForm AI", str(exc))

    def track_action():
        return run_tracking(
            mode=mode_var.get(),
            exercise=exercise_var.get(),
            source_type=source_var.get(),
            video_path=chosen_video(),
        )

    def pinn_action():
        return run_realtime_pinn(
            mode=mode_var.get(),
            exercise=exercise_var.get(),
            source_type=source_var.get(),
            video_path=chosen_video(),
        )

    controls = [
        ("1. Track / Collect Data", lambda: run_with_dashboard(track_action)),
        ("2. Train PINN Model", lambda: run_with_dashboard(train_pinn)),
        ("3. Run Trained PINN Prediction", lambda: run_with_dashboard(pinn_action)),
        ("4. Generate Evaluation Graphs", lambda: run_with_dashboard(evaluate_model)),
        ("5. View Last Graphs", lambda: show_graph_viewer()),
        ("6. Open Results Folder", lambda: open_path(RESULTS_DIR)),
        ("7. Open Exercise Config", lambda: open_path(CONFIG_PATH)),
        ("8. Project Status", lambda: messagebox.showinfo("Project Status", project_status_text())),
    ]

    for index, (label, command) in enumerate(controls):
        row = index // 2
        col = index % 2

        ttk.Button(
            buttons,
            text=label,
            command=command,
            width=34,
        ).grid(
            row=row,
            column=col,
            padx=12,
            pady=12,
            sticky="ew",
        )

    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)

    ttk.Label(
        root,
        text=(
            "Academic prototype. Rehabilitation mode supports movement monitoring "
            "and is not a medical diagnostic system."
        ),
        justify="center",
    ).pack(pady=10)

    ttk.Button(
        root,
        text="Exit",
        command=root.destroy,
    ).pack(pady=10)

    root.mainloop()


def master_menu():
    while True:
        print("\n")
        print("=" * 66)
        print(" BIOFORM AI - PHYSICS-INFORMED BIOMECHANICAL TRACKING ")
        print("=" * 66)
        print("1 - Biomechanical Tracking / Collect Data")
        print("2 - Train PINN Model")
        print("3 - Run Trained PINN Prediction")
        print("4 - Generate Evaluation Graphs")
        print("5 - Show Project Status")
        print("6 - Open GUI Dashboard")
        print("0 - Exit")

        choice = input("\nChoose option: ").strip()

        try:
            if choice == "1":
                run_tracking()

            elif choice == "2":
                train_pinn()

            elif choice == "3":
                run_realtime_pinn()

            elif choice == "4":
                evaluate_model()

            elif choice == "5":
                print("\n" + project_status_text())

            elif choice == "6":
                launch_dashboard()

            elif choice == "0":
                print("\nProgram closed.")
                break

            else:
                print("Invalid option. Choose 0-6.")

        except KeyboardInterrupt:
            print("\nOperation stopped by user.")
            cv2.destroyAllWindows()

        except Exception as exc:
            print("\nERROR:", exc)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        master_menu()
    else:
        launch_dashboard()

