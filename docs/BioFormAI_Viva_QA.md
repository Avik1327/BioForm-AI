# BioForm AI — Viva Questions and Answers

**1. What is the main aim of your project?**  
To estimate exercise form and fatigue from ordinary RGB video using computer vision, biomechanical features and a Physics-Informed Neural Network.

**2. Why did you use MediaPipe?**  
MediaPipe Pose provides body landmarks from RGB frames, which can be converted into joint angles and movement features.

**3. What are the main biomechanical features?**  
Joint angles, ROM, movement velocity, bilateral symmetry, repetitions, effort and time.

**4. What is form tracking?**  
It is the continuous assessment of how correctly a movement is performed using biomechanical rules such as ROM, alignment, symmetry and movement control.

**5. Why use a PINN instead of only a normal neural network?**  
The PINN combines data fitting with a physical fatigue-dynamics constraint, so training is influenced by both observed targets and the selected physics relationship.

**6. What physics equation is used?**  
`dF/dt = αE(t) − βF(t)`, where F is normalized fatigue, E is movement effort, α represents fatigue accumulation and β represents recovery/decay.

**7. What is the total loss?**  
`Ltotal = Ldata + λLphysics`. The current code uses λ = 0.20.

**8. Why do you use MAE and MSE?**  
Fatigue prediction is regression. MAE gives the average absolute prediction error, while MSE penalizes larger errors more strongly.

**9. What are RMSE and R²?**  
RMSE is the square root of MSE and is easier to interpret on the target scale. R² measures how much variation in the test target is explained relative to predicting the test-set mean.

**10. What is your accuracy?**  
This is a regression problem, so formal classification accuracy is not used. A previous run had MAE ≈ 1.95 percentage points and MSE 0.000902 against the current proxy target. The final system also reports RMSE and R².

**11. Can you say the model is 98.05% accurate?**  
No. `100 − 1.95` can only be described as an informal closeness value; it is not a formal regression accuracy metric.

**12. What is your current fatigue ground truth?**  
The current training target is a biomechanical fatigue proxy. RPE can be recorded as metadata, but it is not automatically treated as clinical ground truth.

**13. Why is session-level train/test splitting important?**  
It reduces leakage caused by putting frames from the same video into both training and testing.

**14. What does the website do?**  
It accepts video upload or browser recording, analyzes movement, displays annotated output, session summary and graphs, and provides PINN training, metrics, history and calibration pages.

**15. Is it real-time?**  
The original desktop version supports live webcam processing. The web version records/uploads a clip and analyzes it locally. It is near-session analysis rather than low-latency streaming.

**16. Why do graphs differ for each video?**  
They are generated from the current session's own frame-level data rather than from one global evaluation file.

**17. How is repetition counting performed?**  
Each exercise has low/high joint-angle thresholds and phase transitions. A repetition is counted only when the expected phase transition occurs.

**18. What exercises are supported?**  
The current configuration contains 33 gym and rehabilitation-support movements.

**19. What is the role of symmetry?**  
The absolute left-right joint-angle difference is used as a movement-quality feature and contributes to form feedback.

**20. Is this a medical diagnostic system?**  
No. Rehabilitation mode is movement-monitoring support only. Clinical use would require validated data and clinical testing.

**21. What is the biggest limitation?**  
The fatigue target is still a proxy and single-camera pose estimation is sensitive to camera view and occlusion.

**22. What is your contribution?**  
The contribution is the integrated pipeline: pose-based biomechanics, configurable exercise/form analysis, physics-informed fatigue estimation and web-based session visualization without wearables.

**23. What remains before research-grade validation?**  
Multi-user data collection, empirical threshold calibration, stronger fatigue labels and held-out validation.

**24. Why use Flask?**  
The project backend is already Python, so Flask provides a simple way to expose the same processing and PINN functions through web pages and routes.

**25. What would you add next?**  
Automatic exercise recognition, validated fatigue labels, personalized calibration and secure deployment with user progress tracking.
