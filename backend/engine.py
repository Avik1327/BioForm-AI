from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import math
import os
import re
import shutil
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

# Mutable runtime files can be redirected to a persistent disk in production.
# Local development keeps the original project-folder behaviour.
STORAGE_ROOT = Path(
    os.environ.get("BIOFORM_STORAGE_ROOT", str(ROOT))
).expanduser().resolve()

DATA_DIR = STORAGE_ROOT / "data"
MODEL_DIR = STORAGE_ROOT / "models"
RESULTS_DIR = STORAGE_ROOT / "results"
SESSIONS_DIR = RESULTS_DIR / "sessions"
UPLOAD_RUNTIME_DIR = STORAGE_ROOT / "uploads"

BUNDLED_CONFIG_PATH = ROOT / "exercise_config.json"
BUNDLED_MODEL_PATH = ROOT / "models" / "pinn_fatigue.pt"

CONFIG_PATH = STORAGE_ROOT / "exercise_config.json"
DATA_PATH = DATA_DIR / "biomechanical_dataset.csv"
MODEL_PATH = MODEL_DIR / "pinn_fatigue.pt"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
EVALUATION_PATH = RESULTS_DIR / "evaluation.csv"

for folder in (
    STORAGE_ROOT,
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    SESSIONS_DIR,
    UPLOAD_RUNTIME_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)

# Bootstrap editable config/model into external storage when a persistent
# storage root is used. This also makes /tmp-based demo deploys work.
if CONFIG_PATH != BUNDLED_CONFIG_PATH and not CONFIG_PATH.exists():
    shutil.copy2(BUNDLED_CONFIG_PATH, CONFIG_PATH)

if (
    MODEL_PATH != BUNDLED_MODEL_PATH
    and not MODEL_PATH.exists()
    and BUNDLED_MODEL_PATH.exists()
):
    shutil.copy2(BUNDLED_MODEL_PATH, MODEL_PATH)


def load_exercise_configs() -> dict[str, dict[str, Any]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("exercise_config.json is empty or invalid")
    return data


def save_exercise_configs(configs: dict[str, dict[str, Any]]) -> None:
    CONFIG_PATH.write_text(json.dumps(configs, indent=2), encoding="utf-8")


def names_for_mode(mode: str) -> list[str]:
    configs = load_exercise_configs()
    mode = mode.upper()
    if mode == "REHABILITATION":
        return [n for n, c in configs.items() if c.get("category") == "Rehabilitation"]
    return [n for n, c in configs.items() if c.get("category") != "Rehabilitation"]


def calculate_angle(a, b, c) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


JOINT_KEYS = {
    "knee": ("left_knee_angle", "right_knee_angle"),
    "hip": ("left_hip_angle", "right_hip_angle"),
    "elbow": ("left_elbow_angle", "right_elbow_angle"),
    "shoulder": ("left_shoulder_angle", "right_shoulder_angle"),
    "ankle": ("left_ankle_angle", "right_ankle_angle"),
}


def joint_average(angles: dict[str, float], joint: str) -> float:
    l, r = JOINT_KEYS[joint]
    return (float(angles[l]) + float(angles[r])) / 2.0


def joint_symmetry(angles: dict[str, float], joint: str) -> float:
    l, r = JOINT_KEYS[joint]
    return abs(float(angles[l]) - float(angles[r]))


def normalized_effort(velocity: float, rom: float, symmetry: float, reps: int) -> float:
    v = min(abs(float(velocity)) / 180.0, 1.0)
    r = min(abs(float(rom)) / 120.0, 1.0)
    s = min(abs(float(symmetry)) / 30.0, 1.0)
    rp = min(float(reps) / 20.0, 1.0)
    return float(np.clip(0.45*v + 0.30*r + 0.15*(1.0-s) + 0.10*rp, 0, 1))


def form_status(score: float, symmetry: float) -> str:
    if symmetry > 20:
        return "ASYMMETRY"
    if score >= 85:
        return "GOOD FORM"
    if score >= 65:
        return "ACCEPTABLE"
    return "CHECK FORM"


def exercise_specific_form(exercise: str, angles: dict[str, float], rom: float, velocity: float,
                           symmetry: float, reps: int) -> tuple[float, str, str]:
    cfg = load_exercise_configs()[exercise]
    score = 100.0 - min(float(symmetry) * 1.25, 30.0)
    feedback: list[str] = []
    if velocity > 300:
        score -= 10
        feedback.append("Use more controlled movement")
    if symmetry > 15:
        score -= 10
        feedback.append("Improve left-right symmetry")

    target_rom = 0.0 if cfg.get("static") else abs(float(cfg.get("high", 0))-float(cfg.get("low", 0)))
    if reps > 0 and target_rom and rom < target_rom * 0.65:
        score -= 12
        feedback.append("Increase movement range")

    knee = joint_average(angles, "knee")
    hip = joint_average(angles, "hip")
    elbow = joint_average(angles, "elbow")
    shoulder = joint_average(angles, "shoulder")

    if exercise in {"Squat", "Sumo Squat", "Lunge", "Reverse Lunge", "Bulgarian Split Squat", "Step Up"}:
        if reps > 0 and rom < max(target_rom * 0.70, 25):
            score -= 8
            feedback.append("Check squat/lunge depth")
    elif exercise in {"Bicep Curl", "Hammer Curl", "Elbow Flexion"}:
        if shoulder > 65:
            score -= 10
            feedback.append("Keep upper arm more stable")
    elif exercise in {"Lateral Raise", "Front Raise", "Shoulder Raise", "Shoulder Flexion", "Shoulder Abduction"}:
        if elbow < 115:
            score -= 8
            feedback.append("Avoid excessive elbow bending")
    elif exercise in {"Push Up", "Incline Push Up"}:
        if hip < 145:
            score -= 15
            feedback.append("Keep trunk more aligned")
    elif exercise in {"Deadlift", "Romanian Deadlift"}:
        if knee < 105:
            score -= 8
            feedback.append("Check hip-hinge pattern")
    elif exercise in {"Plank", "Side Plank"}:
        if not 150 <= hip <= 180:
            score -= 20
            feedback.append("Keep trunk and hip aligned")

    if cfg.get("category") == "Rehabilitation":
        if symmetry > 12:
            score -= 8
        if velocity > 240:
            score -= 6
            feedback.append("Use controlled rehabilitation movement")

    score = float(np.clip(score, 0, 100))
    if not feedback:
        feedback.append("Movement quality looks stable")
    return score, form_status(score, symmetry), " | ".join(feedback[:2])


class ExerciseAnalyzer:
    def __init__(self, exercise: str):
        self.exercise = exercise
        self.config = load_exercise_configs()[exercise]
        self.reps = 0
        self.phase = "START"
        self.static_hold_seconds = 0.0

    def update(self, angles: dict[str, float], dt: float = 0.0) -> tuple[int, str, float]:
        cfg = self.config
        active = joint_average(angles, cfg["joint"])
        if cfg.get("static", False):
            self.phase = "HOLD"
            self.static_hold_seconds += max(float(dt), 0)
            return self.reps, self.phase, active
        previous = self.phase
        if active <= cfg["low"]:
            self.phase = cfg["low_phase"]
        elif active >= cfg["high"]:
            self.phase = cfg["high_phase"]
        if cfg.get("rep_on", "high") == "high":
            if previous == cfg["low_phase"] and self.phase == cfg["high_phase"]:
                self.reps += 1
        elif previous == cfg["high_phase"] and self.phase == cfg["low_phase"]:
            self.reps += 1
        return self.reps, self.phase, active


class FatiguePINN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.log_alpha = nn.Parameter(torch.tensor(-2.0))
        self.log_beta = nn.Parameter(torch.tensor(-2.0))

    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def beta(self): return torch.exp(self.log_beta)
    def forward(self, x): return self.network(x)

    def physics_residual(self, x, time_index: int, effort_index: int):
        fatigue = self.forward(x)
        grad = torch.autograd.grad(
            outputs=fatigue, inputs=x, grad_outputs=torch.ones_like(fatigue),
            create_graph=True, retain_graph=True
        )[0]
        dF_dt = grad[:, time_index:time_index+1]
        effort = x[:, effort_index:effort_index+1]
        return dF_dt - (self.alpha * effort - self.beta * fatigue)


FEATURES = [
    "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle",
    "left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle",
    "left_ankle_angle", "right_ankle_angle", "knee_rom", "hip_rom",
    "movement_velocity", "symmetry_difference", "effort", "repetitions", "time",
]
STANDARDIZED_FEATURES = [f for f in FEATURES if f not in {"effort", "time"}]


def safe_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(str(text)).stem).strip("_")
    return s or "session"


def create_session_id(exercise: str, tag: str = "web") -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}_{safe_name(exercise)}_{tag}"


def _extract_angles(landmarks, mp_pose) -> dict[str, float]:
    def p(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y], dtype=np.float32)
    P = mp_pose.PoseLandmark
    ls, rs = p(P.LEFT_SHOULDER.value), p(P.RIGHT_SHOULDER.value)
    le, re = p(P.LEFT_ELBOW.value), p(P.RIGHT_ELBOW.value)
    lw, rw = p(P.LEFT_WRIST.value), p(P.RIGHT_WRIST.value)
    lh, rh = p(P.LEFT_HIP.value), p(P.RIGHT_HIP.value)
    lk, rk = p(P.LEFT_KNEE.value), p(P.RIGHT_KNEE.value)
    la, ra = p(P.LEFT_ANKLE.value), p(P.RIGHT_ANKLE.value)
    lf, rf = p(P.LEFT_FOOT_INDEX.value), p(P.RIGHT_FOOT_INDEX.value)
    return {
        "left_knee_angle": calculate_angle(lh, lk, la), "right_knee_angle": calculate_angle(rh, rk, ra),
        "left_hip_angle": calculate_angle(ls, lh, lk), "right_hip_angle": calculate_angle(rs, rh, rk),
        "left_elbow_angle": calculate_angle(ls, le, lw), "right_elbow_angle": calculate_angle(rs, re, rw),
        "left_shoulder_angle": calculate_angle(lh, ls, le), "right_shoulder_angle": calculate_angle(rh, rs, re),
        "left_ankle_angle": calculate_angle(lk, la, lf), "right_ankle_angle": calculate_angle(rk, ra, rf),
    }


def load_model_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = FatiguePINN(len(checkpoint["features"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {
        "model": model,
        "features": checkpoint["features"],
        "standardized_features": checkpoint["standardized_features"],
        "mean": np.asarray(checkpoint["scaler_mean"]),
        "scale": np.asarray(checkpoint["scaler_scale"]),
        "time_min": float(checkpoint["time_min"]),
        "time_max": float(checkpoint["time_max"]),
    }


def _predict_fatigue(bundle: dict[str, Any], raw: dict[str, float]) -> float:
    x = np.array([[raw[f] for f in bundle["features"]]], dtype=np.float32)
    for j, f in enumerate(bundle["features"]):
        if f in bundle["standardized_features"]:
            idx = bundle["standardized_features"].index(f)
            x[0, j] = (x[0, j] - bundle["mean"][idx]) / max(bundle["scale"][idx], 1e-8)
    with torch.no_grad():
        return float(bundle["model"](torch.tensor(x, dtype=torch.float32)).item() * 100)


def _build_summary(df: pd.DataFrame, exercise: str, mode: str, source: str,
                   fatigue_col: str, fatigue_source: str, rpe: float | None) -> dict[str, Any]:
    def avg(c):
        s = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce").dropna()
        return float(s.mean()) if len(s) else 0.0
    def maxv(c):
        s = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce").dropna()
        return float(s.max()) if len(s) else 0.0
    duration = maxv("time") - float(df["time"].iloc[0]) if len(df) else 0.0
    return {
        "exercise": exercise, "mode": mode, "source": source,
        "duration_seconds": round(max(duration, 0), 2),
        "total_repetitions": int(maxv("repetitions")),
        "static_hold_seconds": round(maxv("static_hold_seconds"), 2),
        "average_rom_deg": round(avg("active_rom"), 2),
        "maximum_rom_deg": round(maxv("active_rom"), 2),
        "average_velocity_deg_s": round(avg("movement_velocity"), 2),
        "average_symmetry_difference_deg": round(avg("symmetry_difference"), 2),
        "average_form_score_percent": round(avg("form_score"), 2),
        "average_fatigue_percent": round(avg(fatigue_col), 2),
        "peak_fatigue_percent": round(maxv(fatigue_col), 2),
        "fatigue_source": fatigue_source,
        "session_rpe_0_to_10": rpe,
        "rpe_equivalent_percent_metadata": None if rpe is None else round(rpe * 10.0, 1),
        "medical_disclaimer": "Movement-monitoring support only; not a medical diagnosis.",
    }


def _save_graphs(df: pd.DataFrame, session_dir: Path, exercise: str, fatigue_col: str) -> list[str]:
    if df.empty:
        return []
    t = df["time"] - float(df["time"].iloc[0])
    specs = [
        (fatigue_col, "Fatigue vs Time", "Fatigue (%)", "fatigue.png"),
        ("form_score", "Form Score vs Time", "Form Score (%)", "form.png"),
        ("active_rom", "ROM vs Time", "ROM (degrees)", "rom.png"),
        ("symmetry_difference", "Symmetry Difference vs Time", "Difference (degrees)", "symmetry.png"),
        ("movement_velocity", "Movement Velocity vs Time", "Velocity (deg/s)", "velocity.png"),
    ]
    saved = []
    for col, title, ylabel, filename in specs:
        if col not in df:
            continue
        plt.figure(figsize=(9, 4.8))
        plt.plot(t, df[col])
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{title} - {exercise}")
        plt.tight_layout()
        path = session_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        saved.append(filename)
    return saved



def _make_browser_friendly_mp4(path: Path) -> None:
    """
    Convert OpenCV's MP4 output to H.264/yuv420p when ffmpeg is available.
    This improves playback compatibility in Safari/Chrome.
    The local desktop version still works when ffmpeg is absent.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not path.exists():
        return

    temp = path.with_name(path.stem + "_h264.mp4")
    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(path),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "24",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(temp),
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=300,
        )
        if completed.returncode == 0 and temp.exists() and temp.stat().st_size > 0:
            path.unlink(missing_ok=True)
            temp.replace(path)
        else:
            temp.unlink(missing_ok=True)
    except Exception:
        temp.unlink(missing_ok=True)


def analyze_video(video_path: str | Path, exercise: str, mode: str = "GYM",
                  use_pinn: bool = True, rpe: float | None = None) -> dict[str, Any]:
    """Headless website analysis. It never opens cv2.imshow windows."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is not installed. Run: pip install -r requirements.txt") from exc

    configs = load_exercise_configs()
    if exercise not in configs:
        raise ValueError(f"Unknown exercise: {exercise}")
    cfg = configs[exercise]
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    session_id = create_session_id(exercise)
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    out_video = session_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    analyzer = ExerciseAnalyzer(exercise)
    bundle = load_model_bundle() if use_pinn else None
    fatigue_source = "PINN model" if bundle else "biomechanical proxy"
    proxy = 0.0
    previous_time = None
    previous_angle = None
    active_hist, knee_hist, hip_hist = [], [], []
    rows: list[dict[str, Any]] = []
    mp_pose = mp.solutions.pose
    drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / float(fps)
            frame_index += 1
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                cv2.putText(frame, "No pose detected - keep full body visible", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                writer.write(frame)
                continue

            angles = _extract_angles(result.pose_landmarks.landmark, mp_pose)
            dt = 0.0 if previous_time is None else max(timestamp - previous_time, 0)
            reps, phase, active_angle = analyzer.update(angles, dt)
            active_hist.append(active_angle)
            knee_hist.append(joint_average(angles, "knee"))
            hip_hist.append(joint_average(angles, "hip"))
            velocity = 0.0 if previous_angle is None or dt <= 1e-8 else abs(active_angle - previous_angle)/dt
            previous_time, previous_angle = timestamp, active_angle
            rom = max(active_hist)-min(active_hist)
            knee_rom = max(knee_hist)-min(knee_hist)
            hip_rom = max(hip_hist)-min(hip_hist)
            symmetry = joint_symmetry(angles, cfg["symmetry_joint"])
            effort = normalized_effort(velocity, rom, symmetry, reps)
            proxy = float(np.clip(proxy*0.985 + effort*1.8, 0, 100))
            score, status, feedback = exercise_specific_form(exercise, angles, rom, velocity, symmetry, reps)

            fatigue = proxy
            if bundle:
                norm_time = np.clip((timestamp-bundle["time_min"])/max(bundle["time_max"]-bundle["time_min"],1e-8), 0, 1)
                raw = {**angles, "knee_rom": knee_rom, "hip_rom": hip_rom,
                       "movement_velocity": velocity, "symmetry_difference": symmetry,
                       "effort": effort, "repetitions": reps, "time": norm_time}
                fatigue = _predict_fatigue(bundle, raw)

            row = {
                "session_id": session_id, "time": timestamp, "mode": mode, "exercise": exercise,
                "source_name": path.name, **angles, "active_joint": cfg["joint"],
                "active_angle": active_angle, "active_rom": rom, "knee_rom": knee_rom,
                "hip_rom": hip_rom, "movement_velocity": velocity, "symmetry_difference": symmetry,
                "effort": effort, "repetitions": reps, "static_hold_seconds": analyzer.static_hold_seconds,
                "form_score": score, "form_status": status, "feedback": feedback,
                "fatigue_score": proxy, "predicted_fatigue": fatigue,
                "session_rpe": rpe,
            }
            rows.append(row)

            drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            panel = frame.copy()
            cv2.rectangle(panel, (10,10), (min(width-10, 660), min(height-10, 330)), (0,0,0), -1)
            frame = cv2.addWeighted(panel, 0.65, frame, 0.35, 0)
            lines = [
                f"BioForm AI | {exercise} | {phase}", f"Reps: {reps}",
                f"{cfg['joint'].title()} angle: {active_angle:.1f} deg | ROM: {rom:.1f} deg",
                f"Velocity: {velocity:.1f} deg/s | Symmetry: {symmetry:.1f} deg",
                f"Form: {score:.0f}% - {status}", f"Fatigue: {fatigue:.1f}% ({fatigue_source})",
                f"Feedback: {feedback[:55]}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (24, 42+i*38), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                            (255,255,255), 2, cv2.LINE_AA)
            writer.write(frame)

    cap.release(); writer.release()
    _make_browser_friendly_mp4(out_video)
    if not rows:
        raise RuntimeError("No usable pose frames were detected in this video")

    df = pd.DataFrame(rows)
    df.to_csv(session_dir/"session.csv", index=False)
    # Append to the training dataset. The model's current supervised target remains the proxy.
    if DATA_PATH.exists():
        old = pd.read_csv(DATA_PATH)
        pd.concat([old, df], ignore_index=True).to_csv(DATA_PATH, index=False)
    else:
        df.to_csv(DATA_PATH, index=False)

    fatigue_col = "predicted_fatigue" if bundle else "fatigue_score"
    graphs = _save_graphs(df, session_dir, exercise, fatigue_col)
    summary = _build_summary(df, exercise, mode, path.name, fatigue_col, fatigue_source, rpe)
    summary["session_id"] = session_id
    summary["graphs"] = graphs
    summary["annotated_video"] = "annotated.mp4"
    (session_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("No collected dataset found. Analyze at least two videos first.")
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + ["fatigue_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Training dataset is missing columns: {missing}")
    if "session_id" not in df:
        raise ValueError("Dataset has no session_id. Recollect data with the web version.")
    df = df[required + ["session_id"]].dropna().reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("Not enough data. Collect more video frames before training.")
    df["fatigue_score"] = np.clip(df["fatigue_score"], 0, 100)/100.0
    return df


def _transform(df: pd.DataFrame, scaler: StandardScaler, tmin: float, tmax: float) -> pd.DataFrame:
    out = df.copy()
    out[STANDARDIZED_FEATURES] = scaler.transform(df[STANDARDIZED_FEATURES])
    out["time"] = (df["time"]-tmin)/max(tmax-tmin,1e-8)
    out["effort"] = np.clip(df["effort"],0,1)
    return out


def train_pinn(epochs: int = 1000) -> dict[str, Any]:
    torch.manual_seed(42); np.random.seed(42)
    df = load_training_data()
    if df["session_id"].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(splitter.split(df, groups=df["session_id"]))
        train_df, test_df = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
        split_method = "session-level GroupShuffleSplit"
    else:
        train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
        split_method = "random fallback (only one session available)"

    scaler = StandardScaler().fit(train_df[STANDARDIZED_FEATURES])
    tmin, tmax = float(train_df["time"].min()), float(train_df["time"].max())
    tr, te = _transform(train_df, scaler, tmin, tmax), _transform(test_df, scaler, tmin, tmax)
    device = torch.device("mps") if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else torch.device("cpu")
    Xtr = torch.tensor(tr[FEATURES].values, dtype=torch.float32, device=device)
    ytr = torch.tensor(tr["fatigue_score"].values.reshape(-1,1), dtype=torch.float32, device=device)
    Xte = torch.tensor(te[FEATURES].values, dtype=torch.float32, device=device)
    yte = torch.tensor(te["fatigue_score"].values.reshape(-1,1), dtype=torch.float32, device=device)
    model = FatiguePINN(len(FEATURES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ti, ei = FEATURES.index("time"), FEATURES.index("effort")
    total_losses, data_losses, physics_losses = [], [], []
    for _ in range(int(epochs)):
        model.train(); x = Xtr.clone().detach().requires_grad_(True)
        pred = model(x); dl = torch.mean((pred-ytr)**2)
        pl = torch.mean(model.physics_residual(x,ti,ei)**2)
        loss = dl + 0.20*pl
        opt.zero_grad(); loss.backward(); opt.step()
        total_losses.append(float(loss.item())); data_losses.append(float(dl.item())); physics_losses.append(float(pl.item()))
    model.eval()
    with torch.no_grad(): pred = model(Xte)
    actual = yte.cpu().numpy().ravel(); predicted = pred.cpu().numpy().ravel()
    mse = float(np.mean((predicted-actual)**2)); mae = float(np.mean(np.abs(predicted-actual)))
    rmse = float(math.sqrt(mse)); r2 = float(r2_score(actual,predicted))
    checkpoint = {"model_state":model.state_dict(),"features":FEATURES,
                  "standardized_features":STANDARDIZED_FEATURES,"scaler_mean":scaler.mean_,
                  "scaler_scale":scaler.scale_,"time_min":tmin,"time_max":tmax,"split_method":split_method}
    torch.save(checkpoint, MODEL_PATH)
    versioned = MODEL_DIR/f"pinn_fatigue_{datetime.now():%Y%m%d_%H%M%S}.pt"; torch.save(checkpoint, versioned)
    pd.DataFrame({"actual_fatigue_percent":actual*100,"predicted_fatigue_percent":predicted*100,
                  "test_session_id":test_df["session_id"].astype(str).values}).to_csv(EVALUATION_PATH,index=False)
    metrics = {"mse":mse,"mae":mae,"rmse":rmse,"r2":r2,"mae_percentage_points":mae*100,
               "split_method":split_method,"training_rows":len(train_df),"testing_rows":len(test_df),
               "training_sessions":int(train_df["session_id"].nunique()),"testing_sessions":int(test_df["session_id"].nunique()),
               "epochs":int(epochs),"alpha":float(model.alpha.item()),"beta":float(model.beta.item())}
    METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    plt.figure(figsize=(9,4.8)); plt.plot(total_losses,label="Total Loss"); plt.plot(data_losses,label="Data Loss"); plt.plot(physics_losses,label="Physics Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("PINN Training Loss"); plt.legend(); plt.tight_layout(); plt.savefig(RESULTS_DIR/"training_loss.png",dpi=160); plt.close()
    generate_evaluation_graphs()
    return metrics


def generate_evaluation_graphs() -> list[str]:
    if not EVALUATION_PATH.exists():
        return []
    df = pd.read_csv(EVALUATION_PATH)
    actual = df["actual_fatigue_percent"].to_numpy(); pred = df["predicted_fatigue_percent"].to_numpy()
    p1 = RESULTS_DIR/"fatigue_prediction.png"
    plt.figure(figsize=(9,4.8)); plt.plot(actual,label="Actual / Dataset"); plt.plot(pred,label="PINN Prediction")
    plt.xlabel("Test Sample"); plt.ylabel("Fatigue (%)"); plt.title("PINN Fatigue Prediction"); plt.legend(); plt.tight_layout(); plt.savefig(p1,dpi=160); plt.close()
    p2 = RESULTS_DIR/"actual_vs_predicted.png"
    plt.figure(figsize=(6,6)); plt.scatter(actual,pred,alpha=.6); lo=float(min(actual.min(),pred.min())); hi=float(max(actual.max(),pred.max())); plt.plot([lo,hi],[lo,hi],linestyle="--")
    plt.xlabel("Actual Fatigue (%)"); plt.ylabel("Predicted Fatigue (%)"); plt.title("Actual vs Predicted"); plt.tight_layout(); plt.savefig(p2,dpi=160); plt.close()
    return [p1.name,p2.name]


def get_metrics() -> dict[str, Any] | None:
    if not METRICS_PATH.exists(): return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def list_sessions() -> list[dict[str, Any]]:
    rows = []
    for d in sorted(SESSIONS_DIR.iterdir(), reverse=True) if SESSIONS_DIR.exists() else []:
        if not d.is_dir(): continue
        sp = d/"summary.json"
        if not sp.exists(): continue
        try: rows.append(json.loads(sp.read_text(encoding="utf-8")))
        except Exception: pass
    return rows


def get_session(session_id: str) -> dict[str, Any]:
    safe = safe_name(session_id)
    path = SESSIONS_DIR/safe/"summary.json"
    if not path.exists(): raise FileNotFoundError(session_id)
    return json.loads(path.read_text(encoding="utf-8"))
