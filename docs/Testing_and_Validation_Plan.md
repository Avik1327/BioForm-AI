# Testing and Validation Plan

## Coding validation
- Run `python -m pytest -q` after installing requirements.
- Verify upload route, session creation, graph generation, model training and metrics page.

## Exercise validation
For Squat, Bicep Curl, Lunge, Push Up and Shoulder Press, record at least 3 sessions each from different users or camera conditions. Manually count reps and compare to system counts. Inspect 20–30 sampled frames for joint-angle plausibility and form feedback.

## Fatigue validation
The current `fatigue_score` is a proxy and is not clinical ground truth. For stronger validation, collect one of the following alongside each session: RPE, validated fatigue questionnaire, physiological measurement, or expert-labelled fatigue. Keep subject/session separation between train and test.

## Acceptance checklist
- Rep-count error documented per exercise.
- No train/test leakage across the same session.
- MAE, MSE, RMSE and R² reported on held-out sessions.
- Video-specific graphs differ between different input videos.
- Limitations and non-diagnostic status stated.
