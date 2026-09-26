from backend.engine import calculate_angle, ExerciseAnalyzer, exercise_specific_form

def angles_for(joint, left, right):
    d = {"left_knee_angle":170,"right_knee_angle":170,"left_hip_angle":170,"right_hip_angle":170,
         "left_elbow_angle":170,"right_elbow_angle":170,"left_shoulder_angle":30,"right_shoulder_angle":30,
         "left_ankle_angle":120,"right_ankle_angle":120}
    d[f"left_{joint}_angle"] = left; d[f"right_{joint}_angle"] = right
    return d

def test_angle_right_angle():
    assert abs(calculate_angle((1,0),(0,0),(0,1))-90) < 1e-3

def test_squat_rep_counter():
    a = ExerciseAnalyzer("Squat")
    a.update(angles_for("knee",100,100))
    reps, phase, _ = a.update(angles_for("knee",165,165))
    assert reps == 1 and phase == "UP"

def test_good_form_score():
    score, status, _ = exercise_specific_form("Squat", angles_for("knee",160,160), 70, 80, 2, 3)
    assert score >= 80 and status in {"GOOD FORM","ACCEPTABLE"}
