from __future__ import annotations

from pathlib import Path
import os

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from backend.engine import (
    CONFIG_PATH, RESULTS_DIR, SESSIONS_DIR, STORAGE_ROOT, UPLOAD_RUNTIME_DIR, analyze_video, generate_evaluation_graphs,
    get_metrics, get_session, list_sessions, load_exercise_configs, names_for_mode,
    save_exercise_configs, train_pinn,
)

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = UPLOAD_RUNTIME_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED = {"mp4", "mov", "avi", "mkv", "m4v", "webm"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("BIOFORM_SECRET", "bioform-ai-local-project")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("BIOFORM_MAX_UPLOAD_MB", "150")) * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED


@app.errorhandler(413)
def upload_too_large(_error):
    max_mb = os.environ.get("BIOFORM_MAX_UPLOAD_MB", "150")
    flash(f"Video is too large. Maximum upload size is {max_mb} MB.", "danger")
    return redirect(url_for("analyze"))


@app.get("/")
def home():
    return render_template("index.html", metrics=get_metrics(), sessions=list_sessions()[:3])


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    mode = request.form.get("mode", "GYM").upper() if request.method == "POST" else "GYM"
    if request.method == "POST":
        exercise = request.form.get("exercise", "")
        use_pinn = request.form.get("use_pinn") == "on"
        rpe_raw = request.form.get("rpe", "").strip()
        rpe = None
        if rpe_raw:
            try:
                rpe = float(rpe_raw)
                if not 0 <= rpe <= 10: raise ValueError
            except ValueError:
                flash("RPE must be between 0 and 10.", "danger")
                return redirect(url_for("analyze"))
        file = request.files.get("video")
        if not file or not file.filename:
            flash("Choose a video file.", "danger")
            return redirect(url_for("analyze"))
        if not allowed_file(file.filename):
            flash("Unsupported video format.", "danger")
            return redirect(url_for("analyze"))
        filename = secure_filename(file.filename)
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)
        try:
            summary = analyze_video(upload_path, exercise, mode=mode, use_pinn=use_pinn, rpe=rpe)
        except Exception as exc:
            flash(str(exc), "danger")
            return redirect(url_for("analyze"))
        finally:
            upload_path.unlink(missing_ok=True)
        return redirect(url_for("session_result", session_id=summary["session_id"]))
    return render_template("analyze.html", modes=["GYM","REHABILITATION"], exercises=names_for_mode("GYM"))


@app.post("/api/analyze-recording")
def analyze_recording():
    file = request.files.get("video")
    exercise = request.form.get("exercise", "Squat")
    mode = request.form.get("mode", "GYM").upper()
    rpe_raw = request.form.get("rpe", "").strip()
    rpe = float(rpe_raw) if rpe_raw else None
    use_pinn = request.form.get("use_pinn", "true").lower() == "true"
    if not file:
        return jsonify({"error":"No recording received"}), 400
    filename = secure_filename(file.filename or "recording.webm")
    path = UPLOAD_DIR/filename; file.save(path)
    try:
        summary = analyze_video(path, exercise, mode=mode, use_pinn=use_pinn, rpe=rpe)
        return jsonify({"redirect": url_for("session_result", session_id=summary["session_id"])})
    except Exception as exc:
        return jsonify({"error":str(exc)}), 500
    finally:
        path.unlink(missing_ok=True)


@app.get("/session/<session_id>")
def session_result(session_id):
    try:
        summary = get_session(session_id)
    except FileNotFoundError:
        abort(404)
    return render_template("result.html", summary=summary)


@app.get("/media/<session_id>/<path:filename>")
def session_media(session_id, filename):
    directory = SESSIONS_DIR / session_id
    return send_from_directory(directory, filename)


@app.get("/result-file/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        try:
            epochs = int(request.form.get("epochs", "1000"))
            epochs = max(50, min(5000, epochs))
            metrics = train_pinn(epochs=epochs)
            flash("PINN training completed.", "success")
            return render_template("train.html", metrics=metrics)
        except Exception as exc:
            flash(str(exc), "danger")
    return render_template("train.html", metrics=get_metrics())


@app.get("/sessions")
def sessions():
    return render_template("sessions.html", sessions=list_sessions())


@app.route("/calibration", methods=["GET", "POST"])
def calibration():
    configs = load_exercise_configs()
    selected = request.values.get("exercise") or next(iter(configs))
    if selected not in configs: selected = next(iter(configs))
    if request.method == "POST":
        cfg = configs[selected]
        try:
            if not cfg.get("static", False):
                cfg["low"] = float(request.form["low"])
                cfg["high"] = float(request.form["high"])
            cfg["view"] = request.form.get("view", cfg.get("view","front/side")).strip()
            save_exercise_configs(configs)
            flash(f"Calibration saved for {selected}.", "success")
        except Exception as exc:
            flash(str(exc), "danger")
        return redirect(url_for("calibration", exercise=selected))
    return render_template("calibration.html", configs=configs, selected=selected, cfg=configs[selected])


@app.get("/metrics")
def metrics():
    graphs = generate_evaluation_graphs()
    return render_template("metrics.html", metrics=get_metrics(), graphs=graphs)


@app.get("/api/exercises")
def api_exercises():
    return jsonify(names_for_mode(request.args.get("mode","GYM")))


@app.get("/api/status")
def api_status():
    return jsonify({"model_ready": (STORAGE_ROOT/"models"/"pinn_fatigue.pt").exists(),
                    "metrics": get_metrics(), "session_count": len(list_sessions()),
                    "storage_root": str(STORAGE_ROOT)})


@app.get("/health")
def health():
    return {"status":"ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
