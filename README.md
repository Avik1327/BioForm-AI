# BioForm AI — Premium Web Edition — Final Web Project

Web-based final-year project for **Physics-Informed Neural Networks (PINNs) for Real-Time Biomechanical Fatigue & Form Tracking**.

## Run on macOS in VS Code

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in Chrome/Safari.

## Recommended final workflow
1. Analyze at least 2–5 different exercise videos/sessions.
2. Check video-specific reps, ROM, form, symmetry and graphs.
3. Go to **Calibration** and tune thresholds when necessary.
4. Go to **Train PINN** and train using a session-level split.
5. Record MAE, MSE, RMSE and R² from the Metrics page.
6. Test the trained model on a new video that was not used for training.

## Scientific limitation
The current supervised fatigue target is a **biomechanical proxy**. The optional session RPE is stored as metadata and is not silently substituted as clinical ground truth. Clinical claims require validated fatigue labels and clinical testing.

## Browser camera
The Analyze page can record a clip using `getUserMedia` / `MediaRecorder`, then sends the recorded clip to the same analysis backend. This is browser camera capture, not low-latency frame-by-frame streaming.


## Premium UI refresh

This edition includes a redesigned responsive interface, drag-and-drop upload, animated processing state, premium result cards, circular form/fatigue gauges, session cards, polished graphs, mobile navigation, and an upgraded BioForm AI visual identity. The computer-vision and PINN backend logic is unchanged.
