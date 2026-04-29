# Fight Punch Classification Pipeline

<!--
Add a short GIF of the punch recording / model demo here.

Example:
![Punch recording demo](assets/punch-recording.gif)
-->

## Overview

This project detects punch-like movements in fight videos, stores short event clips with extracted pose-based features, allows manual review of those clips, trains a Random Forest classifier on the reviewed labels, and then evaluates the trained model on detected punch events.

The full pipeline has four main stages:

1. `merge.py` detects candidate punch events from raw fight videos.
2. `review_events.py` lets you manually label each saved event as a punch type or `no_punch`.
3. `classifier.py` trains and saves a Random Forest model using the reviewed feature data.
4. `predict_punches.py` runs detection and classification directly on a video, then writes model predictions next to actual reviewed labels.
5. `eval.py` reads `summary.csv` and calculates performance metrics.

## Pipeline Flow

```text
Raw fight video
    |
    v
merge.py
    - YOLO detects people
    - ByteTrack tracks fighters
    - MediaPipe extracts body landmarks
    - custom geometry detects candidate punch events
    |
    v
data_collection/<video_name>/punch_*/features.json + clip.mp4
    |
    v
review_events.py
    - human labels each clip
    - labels: jab, cross, hook, uppercut, overhand, no_punch, unclear
    |
    v
classifier.py
    - loads reviewed features
    - trains Random Forest
    - saves random_forest_model.joblib
    |
    v
predict_punches.py
    - runs detection on a video
    - feeds each event directly into saved model
    - matches prediction to actual label from data_collection
    |
    v
model_predictions/<video_name>/summary.csv
    |
    v
eval.py
    - accuracy
    - precision / recall / F1
    - confusion matrix
    - punch vs no_punch metrics
```

## Project Files

### `merge.py`

This is the original data collection script. It processes a raw video and saves candidate punch events.

Main responsibilities:

- Opens a fight video from `VIDEO_PATH`.
- Uses YOLO person detection with ByteTrack tracking.
- Locks onto two fighters and attempts to keep fighter IDs stable.
- Runs MediaPipe Pose on fighter crops.
- Tracks shoulder, elbow, wrist, and hip landmarks.
- Calculates punch-related features such as:
  - `duration_frames`
  - `extension_start`
  - `extension_peak`
  - `extension_delta`
  - `relative_wrist_travel`
  - `relative_wrist_speed`
  - `elbow_angle_start`
  - `elbow_angle_peak`
  - `elbow_angle_delta`
  - `toward_score`
  - `opponent_distance_delta`
  - `hip_rotation_delta`
- Saves each detected candidate event under `data_collection`.

Output structure:

```text
data_collection/
  test_video2/
    punch_0001_f1_left_frame52/
      clip.mp4
      features.json
```

Each `features.json` contains metadata, the detected arm/fighter, the frame window, and the extracted feature dictionary.

### `review_events.py`

This script is used after `merge.py`. It opens each saved clip and lets you manually label it.

Keyboard labels:

```text
j = jab
c = cross
h = hook
u = uppercut
n = no_punch
x = unclear
space = pause/play
r = replay
b = back
q = quit
```

When you label a clip, `review_events.py` edits the corresponding `features.json`.

For a real punch:

```json
{
  "punch": true,
  "punch_type": "jab"
}
```

For a false positive:

```json
{
  "punch": false,
  "punch_type": "none"
}
```

When training and evaluating, false positives are treated as:

```text
no_punch
```

### `classifier.py`

This script trains the punch classifier.

It loads reviewed feature records from:

```text
data_collection/test_video*/punch_*/features.json
```

It flattens each JSON file into a tabular row:

```text
fighter, arm, duration_frames, extension_start, ..., hip_rotation_delta, label
```

The label is created like this:

```python
if item["punch"]:
    row["label"] = item["punch_type"]
else:
    row["label"] = "no_punch"
```

The current valid labels are:

```text
jab
cross
hook
uppercut
overhand
no_punch
```

Important note: `classifier.py` was updated so `no_punch` examples are kept in the training data. This matters because false positives should be a real class the model learns, not just something guessed from low confidence.

The trained model is saved as:

```text
random_forest_model.joblib
```

### `predict_punches.py`

This is the direct model testing script.

It combines the detection logic from `merge.py` with the trained model from `classifier.py`.

Instead of only saving unlabeled candidate events, it:

- runs YOLO + ByteTrack + MediaPipe on a video
- detects candidate punches using the same geometry features
- builds a model input row from each event
- loads `random_forest_model.joblib`
- predicts the punch class
- loads the reviewed actual labels from `data_collection/<video_name>`
- matches each prediction to the closest reviewed event using:
  - `fighter`
  - `arm`
  - closest `peak_frame`
- writes a `summary.csv` file for evaluation

Main output:

```text
model_predictions/<video_name>/summary.csv
```

The useful columns are:

```text
peak_frame
fighter
arm
model_prediction
actual_label
confidence
model_label
actual_peak_frame
```

`model_prediction` is the prediction you should compare against `actual_label`.

`actual_label` comes from:

```text
data_collection/<video_name>/punch_*/features.json
```

### `eval.py`

This script reads `summary.csv` and calculates model performance metrics.

It uses:

```text
model_prediction
actual_label
```

from `summary.csv`.

It skips rows where `actual_label` is missing or `unlabeled`.

It prints:

- total rows
- evaluated rows
- accuracy
- macro precision / recall / F1
- weighted precision / recall / F1
- per-class precision / recall / F1 / support
- punch vs `no_punch` metrics

It writes:

```text
metrics.json
confusion_matrix.csv
class_metrics.csv
```

next to the input `summary.csv`.

## How To Run Each Step

Run all commands from the project root:

```bash
cd "/Users/ivancabrilo/Desktop/UofR/Spring 2026/CSC 449/MV_FINAL_PROJECT"
```

## 1. Collect Candidate Punch Events

Edit `VIDEO_PATH` near the top of `merge.py`:

```python
VIDEO_PATH = "video_data/test_video2.mp4"
```

Then run:

```bash
python3 merge.py
```

This will create event folders under:

```text
data_collection/test_video2/
```

Each event folder should contain:

```text
clip.mp4
features.json
```

## 2. Review And Label Events

To review all unlabeled clips:

```bash
python3 review_events.py
```

To review only one video folder:

```bash
python3 review_events.py --video test_video2
```

To include already labeled clips:

```bash
python3 review_events.py --video test_video2 --include-labeled
```

The reviewed labels are saved directly into each event's `features.json`.

## 3. Train The Classifier

After reviewing clips, train the model:

```bash
python3 classifier.py
```

This reads the reviewed `data_collection` records and saves:

```text
random_forest_model.joblib
```

If you add more reviewed data later, rerun `classifier.py` so the model learns from the new labels.

## 4. Run Direct Model Prediction On A Video

To test the trained model on `test_video2`:

```bash
python3 predict_punches.py video_data/test_video2.mp4
```

This creates:

```text
model_predictions/test_video2/
```

The main file for analysis is:

```text
model_predictions/test_video2/summary.csv
```

To run without the OpenCV display window:

```bash
python3 predict_punches.py video_data/test_video2.mp4 --no-display
```

To avoid saving event clips and only save JSON/CSV output:

```bash
python3 predict_punches.py video_data/test_video2.mp4 --no-clips
```

To loosen the frame matching tolerance between detected model events and reviewed actual events:

```bash
python3 predict_punches.py video_data/test_video2.mp4 --label-frame-tolerance 8
```

Default tolerance is 3 frames.

## 5. Evaluate Model Performance

After `summary.csv` exists, run:

```bash
python3 eval.py model_predictions/test_video2/summary.csv
```

This prints performance metrics and writes:

```text
model_predictions/test_video2/metrics.json
model_predictions/test_video2/confusion_matrix.csv
model_predictions/test_video2/class_metrics.csv
```

You can use these files in your report.

## Recommended Full Workflow

For a new video:

```bash
cd "/Users/ivancabrilo/Desktop/UofR/Spring 2026/CSC 449/MV_FINAL_PROJECT"
```

Edit `VIDEO_PATH` in `merge.py`, then run:

```bash
python3 merge.py
```

Review labels:

```bash
python3 review_events.py --video test_video2
```

Retrain the model:

```bash
python3 classifier.py
```

Run direct prediction:

```bash
python3 predict_punches.py video_data/test_video2.mp4
```

Evaluate:

```bash
python3 eval.py model_predictions/test_video2/summary.csv
```

## Important Concepts

### Candidate Event

A candidate event is something the geometry-based detector thinks might be a punch. It is not guaranteed to be a real punch.

That is why `review_events.py` exists: some candidate events are real punches, and some are false positives.

### `no_punch`

`no_punch` means the detector found a candidate event, but human review decided it was not actually a punch.

These examples are important because they teach the model to reject false positives.

### `model_prediction`

This is the model's predicted class for a detected event.

Possible values include:

```text
jab
cross
hook
uppercut
overhand
no_punch
```

### `actual_label`

This is the reviewed ground-truth label from `data_collection`.

It is used to evaluate model performance.

### `confidence`

This is the model's maximum predicted probability for the event. Higher confidence means the model was more certain about its prediction.

## Output Files

### `data_collection`

Generated by `merge.py` and edited by `review_events.py`.

```text
data_collection/<video_name>/punch_*/features.json
data_collection/<video_name>/punch_*/clip.mp4
```

### `random_forest_model.joblib`

Generated by `classifier.py`.

This is the trained model used by `predict_punches.py`.

### `model_predictions`

Generated by `predict_punches.py`.

```text
model_predictions/<video_name>/summary.csv
model_predictions/<video_name>/event_*/prediction.json
model_predictions/<video_name>/event_*/clip.mp4
```

### Evaluation Files

Generated by `eval.py`.

```text
model_predictions/<video_name>/metrics.json
model_predictions/<video_name>/confusion_matrix.csv
model_predictions/<video_name>/class_metrics.csv
```

## Metrics To Include In A Report

Useful metrics from `eval.py`:

- Overall accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted precision
- Weighted recall
- Weighted F1
- Per-class precision
- Per-class recall
- Per-class F1
- Per-class support
- Confusion matrix
- Punch vs `no_punch` precision
- Punch vs `no_punch` recall
- Punch vs `no_punch` F1

Macro metrics treat each class equally.

Weighted metrics account for class imbalance by weighting each class by its support.

The confusion matrix is especially useful for showing which punch types the model confuses with each other.

## Dependencies

The project uses:

- Python
- OpenCV
- MediaPipe
- Ultralytics YOLO
- NumPy
- pandas
- scikit-learn
- joblib

If a package is missing, install the required Python dependencies in the same environment used to run the scripts.

## Notes And Current Limitations

- `merge.py` currently requires manually editing `VIDEO_PATH` to choose the input video.
- `predict_punches.py` accepts the video path as a command-line argument.
- The actual label matching in `predict_punches.py` assumes the reviewed labels already exist in `data_collection/<video_name>`.
- If `actual_label` is blank in `summary.csv`, the prediction did not match a reviewed event within the frame tolerance.
- `unclear` and `unlabeled` events are not used by `classifier.py` because they are not part of the valid training classes.
- Model quality depends heavily on the number and balance of reviewed examples in `data_collection`.
