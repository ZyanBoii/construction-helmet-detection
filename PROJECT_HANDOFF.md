# Construction Helmet Detection Project - Current Handoff

Last updated: 2026-08-28

This is the current source of truth for continuing the university construction
helmet detection project. Inspect referenced files before changing them because
the code and active model path may change after this update.

## 1. Executive Summary

The project is at the functional MVP and teacher-demo preparation stage.

Completed:

- Construction-specific helmet and no_helmet class definitions
- Pascal VOC to YOLO conversion and train/validation/test splits
- Smoke test, 30-epoch baseline, and newer 50-epoch training
- Formal overall and per-class held-out test evaluation
- Saved evaluation plots and prediction images
- Local continuous-webcam alert prototype
- Streamlit upload and browser-camera snapshot application
- Initial public-deployment files

The user reported that current testing is complete and considers the result
acceptable for the university teacher demonstration. Accuracy is not 100
percent. Further training and dataset improvement are intentionally postponed
until after the deadline.

Immediate priority:

1. Improve and stabilize the Streamlit interface.
2. Test the final teacher-demo flow locally.
3. Learn server/deployment development by publishing the Streamlit demo.
4. Return to accuracy improvement after the presentation.

This is an educational prototype, not a production safety control or a
replacement for human supervision.

## 2. Goal and Deadline Scope

The model detects two head-level classes:

- helmet: a visible head wearing a construction safety helmet
- no_helmet: a visible bare head without a construction safety helmet

The deadline version accepts an uploaded image or browser-camera snapshot,
draws detections, counts both classes, and shows a warning when at least one
no_helmet detection is present.

Continuous CCTV, multi-camera processing, tracking, databases, notifications,
user accounts, and a separate API backend are future work.

## 3. Workspace and Important Files

Project root:

    D:\Porgraming_learnig_file\YOLO_learing\yolo helmet

Important files:

- PROJECT_HANDOFF.md
- README.md
- HELMET_ALERT_ROADMAP_MM.md
- prepare_dataset.py
- train_helmet.py
- evaluate_helmet.py
- evaluate_helmet1.py
- webcam_alert.py
- streamlit_app.py
- requirements.txt
- .streamlit/config.toml
- .gitignore

Important folders include archive, construction_helmet_yolo,
helmet_extra.v1i.yolo26, runs, .venv, and Ultralytics.

## 4. Dataset and Classes

The original archive contains 5,000 validated PNG and Pascal VOC XML pairs.

Class mapping:

- helmet becomes class 0, helmet
- head becomes class 1, no_helmet
- person is intentionally ignored because it is sparse and outside the direct
  head-level detector scope

Do not change the class order. Application code expects class ID 1 to mean
no_helmet.

Current converted counts:

| Split | Images | Labels |
|---|---:|---:|
| Train | 3,511 | 3,511 |
| Validation | 1,000 | 1,000 |
| Test | 500 | 500 |

The original conversion produced 3,500 training images. The current split has
11 additional image-label pairs. Preserve them for the deadline demo, but
document their source before future retraining or research reporting.

Dataset configuration:

    construction_helmet_yolo/data.yaml

## 5. Training History

Smoke test:

    runs/smoke_test

The smoke test proves the pipeline can run; its metrics are not final evidence.

30-epoch baseline:

    runs/construction_helmet_yolo26n/weights/best.pt

Final baseline validation row:

- Precision: 0.90357
- Recall: 0.86043
- mAP50: 0.92402
- mAP50-95: 0.59368

Current preferred 50-epoch model:

    runs/construction_helmet_yolo26nExtra/weights/best.pt

It used YOLO26n transfer learning, image size 416, batch size 8, GPU 0,
workers 0, random seed 42, and 50 epochs.

Epoch-50 validation row:

- Precision: 0.91615
- Recall: 0.88100
- mAP50: 0.93240
- mAP50-95: 0.60256

The highest saved validation mAP50-95 row was about 0.60317 at epoch 49.
Validation metrics are not held-out test metrics.

## 6. Verified Held-out Test Results

evaluate_helmet1.py evaluates the current Extra best.pt and saves overall and
per-class reports under:

    runs/evaluation/test_metrics_class_report

Saved evidence includes class_metrics.csv, class_metrics.txt, confusion
matrices, metric curves, and labelled/predicted batches.

Verified overall test metrics from 2026-08-28:

- Precision: 0.89257
- Recall: 0.86479
- mAP50: 0.91459
- mAP50-95: 0.60269

Verified per-class metrics:

| Class | Images | Instances | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| helmet | 456 | 1,832 | 0.91134 | 0.88648 | 0.89874 | 0.93828 | 0.61415 |
| no_helmet | 86 | 580 | 0.87381 | 0.84310 | 0.85818 | 0.89090 | 0.59123 |

The no_helmet recall of about 84.31 percent means some unsafe cases are still
missed. This is acceptable only for the classroom demo, not real enforcement.

## 7. Current Applications

### Streamlit

File:

    streamlit_app.py

Current behavior:

- Loads the 50-epoch Extra best.pt model
- Uses CUDA when available and CPU otherwise
- Caches the model between Streamlit reruns
- Accepts JPG, JPEG, and PNG uploads
- Supports browser-camera snapshots
- Provides a confidence slider
- Draws boxes and labels
- Counts helmet and no_helmet detections
- Shows a warning when no_helmet is detected
- Displays a university-demo disclaimer

The app handles still images and snapshots, not continuous browser video or
direct cloud access to the laptop webcam/CCTV.

Local command:

    .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py

Local URL:

    http://localhost:8501

### Local continuous webcam

File:

    webcam_alert.py

Current settings:

- inference confidence: 0.25
- no-helmet confidence: 0.35
- confirmation time: 0.75 seconds
- alert cooldown: 10 seconds
- camera index: 0

It uses OpenCV and Windows winsound.Beep. It does not track people, save events,
or support a remote CCTV server.

Important: webcam_alert.py still points to the older 30-epoch model while
Streamlit and the latest evaluation use the 50-epoch Extra model. Synchronize
the path before presenting the webcam version.

## 8. User Testing Decision

On 2026-08-28 the user reported testing complete and the result good enough for
the teacher demo even though it is not perfect.

Therefore:

- Do not retrain before the deadline.
- Do not redesign the dataset now.
- Do not add tracking, CCTV infrastructure, or a complex backend now.
- Keep the educational and safety disclaimer visible.
- Record obvious failure cases for later improvement.
- Prioritize a stable, understandable, presentable demo.

## 9. Known Issues

1. webcam_alert.py and streamlit_app.py use different model runs.
2. evaluate_helmet.py has an older output-directory message;
   evaluate_helmet1.py is the authoritative class-report evaluator.
3. evaluate_helmet1.py should receive a clearer name after the deadline.
4. Git has no commits and all project files are currently untracked.
5. Public deployment has not been completed or verified.
6. Poor lighting, blur, occlusion, distance, angle, and domain differences can
   reduce accuracy.
7. The alert does not associate helmets with individual people.

## 10. Deadline-first Streamlit Modification Plan

Implement and verify in this order:

1. Keep one authoritative current-model path.
2. Improve the title, short instructions, and demo disclaimer.
3. Put input and confidence controls in a clear sidebar/control section.
4. Keep upload and browser-camera snapshot options.
5. Display helmet and no_helmet counts side by side.
6. Show a clear green safe panel or red warning panel.
7. Add a compact table with detected class and confidence.
8. Handle invalid images and inference errors with readable messages.
9. Show original and annotated images side by side when space allows.
10. Test a fixed demo set with helmet, no-helmet, mixed, and difficult images.

Do not add authentication, a database, continuous streaming, or a large visual
redesign before the presentation flow is stable.

## 11. Deadline-first Server and Deployment Plan

For the teacher demo, Streamlit itself is the web application server. A separate
Flask or FastAPI backend is not required now.

After the Streamlit UI is final:

1. Run the app locally and complete the final demo checklist.
2. Confirm requirements.txt, .streamlit/config.toml, .gitignore, README,
   streamlit_app.py, and the current best.pt are included.
3. Create the first Git commit.
4. Create an empty GitHub repository.
5. Add the remote and push the project.
6. Create a Streamlit Community Cloud app using the repository, default branch,
   and streamlit_app.py entrypoint.
7. Wait for dependencies and model loading.
8. Test the public URL from another device with non-sensitive images.
9. Save the URL and backup screenshots.
10. Keep a local-demo backup in case presentation internet fails.

Cloud Streamlit supports upload and browser-camera snapshots. It cannot directly
read the laptop's local OpenCV stream or private CCTV. Future live CCTV needs an
edge-camera process plus an API/server.

## 12. After-deadline Backlog

- Review missed no_helmet cases.
- Audit duplicates and document the 11 additional training images.
- Collect target-camera construction images.
- Fine-tune with correctly split target-domain data.
- Compare thresholds using precision, recall, and F1.
- Add saved-video inference.
- Add tracking and per-person alert logic.
- Save alert screenshots, timestamps, and structured events.
- Design an edge-camera plus FastAPI backend for remote CCTV.
- Add authentication, storage, monitoring, and notifications later.

## 13. Deployment Files Already Present

- requirements.txt
- .streamlit/config.toml
- .gitignore
- README.md
- streamlit_app.py
- runs/construction_helmet_yolo26nExtra/weights/best.pt

Pinned dependencies:

- streamlit==1.61.1
- ultralytics==8.4.121

## 14. Next-session Instruction

Read PROJECT_HANDOFF.md and inspect the current files. The model and local
testing are accepted for the deadline, so do not retrain now. Continue with the
deadline-first Streamlit modification plan using the latest Extra best.pt.
Explain changes in Burmese step by step and verify locally. After the UI is
stable, continue with GitHub and Streamlit deployment. Keep it an educational
image-upload and browser-camera demo, not a production safety system.
