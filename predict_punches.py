import argparse
import csv
import json
import math
import os
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from ultralytics import YOLO


MODEL_PATH = "random_forest_model.joblib"
YOLO_MODEL_PATH = "yolo26n.pt"
OUTPUT_DIR = "model_predictions"

SCENE_CUT_THRESHOLD = 0.8
MISSING_FRAMES_RELOCK = 10
SINGLE_FIGHTER_MISSING_UNLOCK = 6
MIN_MATCH_SCORE = 0.5
SIG_UPDATE_WEIGHT = 0.1
INITIAL_LOCK_FRAMES = 20
MIN_SEPARATION_RATIO = 0.18

SCENE_CUT_IGNORE_FRAMES = 5
CLIP_FRAMES_BEFORE = 14
CLIP_FRAMES_AFTER = 12
PUNCH_LOOKBACK_FRAMES = 7
PUNCH_CONFIRM_AFTER = 2
MIN_ARM_VISIBILITY = 0.45
MIN_EXTENSION_DELTA = 0.12
MIN_PEAK_EXTENSION = 0.80
MIN_REL_WRIST_TRAVEL = 0.14
MIN_ABS_WRIST_TRAVEL = 0.16
MIN_REL_WRIST_SPEED = 0.020
MIN_ELBOW_ANGLE_AT_PEAK = 70
SAME_ARM_DEBOUNCE_FRAMES = 10
SMOOTH_ALPHA = 0.50
RAW_FRAME_BUFFER_SIZE = 120
LANDMARK_HISTORY_SIZE = 160

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

ARM_KEYS = {
    "left": {"shoulder": "L_SHOULDER", "elbow": "L_ELBOW", "wrist": "L_WRIST", "hip": "L_HIP"},
    "right": {"shoulder": "R_SHOULDER", "elbow": "R_ELBOW", "wrist": "R_WRIST", "hip": "R_HIP"},
}

EXCLUDED_LANDMARKS = list(range(0, 11)) + list(range(25, 33))
MP_POSE = mp.solutions.pose

BYTETRACK_CONFIG = """tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.5
track_buffer: 60
match_thresh: 0.7
fuse_score: True
"""


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center_x(box):
    x1, _, x2, _ = box
    return (x1 + x2) / 2.0


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def angle_delta_deg(a, b):
    return (a - b + 180) % 360 - 180


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter_area
    return inter_area / union if union > 0 else 0.0


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def contains_center(box_a, box_b):
    x1, y1, x2, y2 = box_a
    cx, cy = box_center(box_b)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def dedupe_detections(detections, overlap_thresh=0.25, center_dist_thresh=90):
    kept = {}
    ordered = sorted(
        detections.items(),
        key=lambda item: (item[1]["conf"], box_area(item[1]["box"])),
        reverse=True,
    )
    for tid, det in ordered:
        box = det["box"]
        is_dup = False
        for kept_det in kept.values():
            kept_box = kept_det["box"]
            if (
                iou(box, kept_box) >= overlap_thresh
                or contains_center(box, kept_box)
                or contains_center(kept_box, box)
                or center_distance(box, kept_box) <= center_dist_thresh
            ):
                is_dup = True
                break
        if not is_dup:
            kept[tid] = det
    return kept


def xy(point):
    return np.array([point[0], point[1]], dtype=np.float32)


def dist_xy(a, b):
    return float(np.linalg.norm(xy(a) - xy(b)))


def unit_vec(v):
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else None


def angle_3pt(a, b, c):
    a, b, c = xy(a), xy(b), xy(c)
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return 0.0
    cos_v = max(-1.0, min(1.0, float(np.dot(v1, v2) / denom)))
    return math.degrees(math.acos(cos_v))


def point_in_expanded_box(point, box, expand_ratio=0.25):
    if box is None:
        return False
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return (
        x1 - bw * expand_ratio <= point[0] <= x2 + bw * expand_ratio
        and y1 - bh * expand_ratio <= point[1] <= y2 + bh * expand_ratio
    )


def torso_crop(img, box):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1 = x1 + int(bw * 0.2)
    tx2 = x2 - int(bw * 0.2)
    ty1 = y1 + int(bh * 0.25)
    ty2 = y2 - int(bh * 0.35)
    return img[max(0, ty1):ty2, max(0, tx1):tx2]


def appearance_signature(img, box):
    crop = torso_crop(img, box)
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def dark_torso_ratio(img, box):
    crop = torso_crop(img, box)
    if crop.size == 0:
        return 1.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    dark_mask = (hsv[:, :, 2] < 75) & (hsv[:, :, 1] < 110)
    return float(np.mean(dark_mask))


def hist_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


def frame_signature(img):
    small = cv2.resize(img, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def fighter_candidate_score(img, box, conf):
    h, w = img.shape[:2]
    area_norm = box_area(box) / float(w * h)
    cx, cy = box_center(box)
    center_x_norm = cx / w
    center_y_norm = cy / h
    bottom_norm = box[3] / h
    top_norm = box[1] / h
    height_norm = (box[3] - box[1]) / h
    return (
        2.2 * area_norm
        + 0.9 * bottom_norm
        + 0.6 * center_y_norm
        + 0.5 * conf
        - abs(center_x_norm - 0.5) * 1.2
        - dark_torso_ratio(img, box) * 0.45
        - max(0.0, 0.42 - center_y_norm) * 3.5
        - max(0.0, 0.18 - top_norm) * 2.5
        - max(0.0, height_norm - 0.62) * 2.5
    )


def get_arm_metrics(snapshot, arm):
    keys = ARM_KEYS[arm]
    shoulder = snapshot[keys["shoulder"]]
    elbow = snapshot[keys["elbow"]]
    wrist = snapshot[keys["wrist"]]
    hip = snapshot[keys["hip"]]
    vis_min = min(shoulder[3], elbow[3], wrist[3], hip[3])
    if vis_min < MIN_ARM_VISIBILITY:
        return None
    torso_len = dist_xy(shoulder, hip)
    if torso_len < 1e-6:
        return None
    return {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "hip": hip,
        "visibility_min": float(vis_min),
        "torso_len": float(torso_len),
        "extension": dist_xy(shoulder, wrist) / torso_len,
        "elbow_angle": angle_3pt(shoulder, elbow, wrist),
        "wrist_relative_to_shoulder": xy(wrist) - xy(shoulder),
        "opponent_center": snapshot.get("opponent_center"),
        "opponent_box": snapshot.get("opponent_box"),
    }


class SimplePunchDetector:
    def __init__(self):
        self.history = deque(maxlen=LANDMARK_HISTORY_SIZE)
        self.last_event_frame = {"left": -10_000, "right": -10_000}
        self.prev_smoothed = None

    def reset(self):
        self.history.clear()
        self.prev_smoothed = None

    def smooth_snapshot(self, raw):
        if self.prev_smoothed is None:
            self.prev_smoothed = raw
            return raw
        smoothed = {
            "frame": raw["frame"],
            "fighter": raw["fighter"],
            "fighter_box": raw["fighter_box"],
            "opponent_box": raw["opponent_box"],
            "opponent_center": raw["opponent_center"],
        }
        for key in ["L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_WRIST", "R_WRIST", "L_HIP", "R_HIP"]:
            cur, prev = raw[key], self.prev_smoothed[key]
            smoothed[key] = (
                float((1 - SMOOTH_ALPHA) * prev[0] + SMOOTH_ALPHA * cur[0]),
                float((1 - SMOOTH_ALPHA) * prev[1] + SMOOTH_ALPHA * cur[1]),
                float((1 - SMOOTH_ALPHA) * prev[2] + SMOOTH_ALPHA * cur[2]),
                float(cur[3]),
            )
        self.prev_smoothed = smoothed
        return smoothed

    def push_landmarks(self, frame_count, fighter_id, landmarks, fighter_box, opponent_box=None):
        x1, y1, x2, y2 = fighter_box
        bw, bh = x2 - x1, y2 - y1
        opp_center = box_center(opponent_box) if opponent_box is not None else None

        def ffp(idx):
            lm = landmarks[idx]
            return (float(lm.x * bw + x1), float(lm.y * bh + y1), float(lm.z * bw), float(lm.visibility))

        raw = {
            "frame": int(frame_count),
            "fighter": int(fighter_id),
            "fighter_box": tuple(map(float, fighter_box)),
            "opponent_box": tuple(map(float, opponent_box)) if opponent_box else None,
            "opponent_center": tuple(map(float, opp_center)) if opp_center else None,
            "L_SHOULDER": ffp(L_SHOULDER),
            "R_SHOULDER": ffp(R_SHOULDER),
            "L_ELBOW": ffp(L_ELBOW),
            "R_ELBOW": ffp(R_ELBOW),
            "L_WRIST": ffp(L_WRIST),
            "R_WRIST": ffp(R_WRIST),
            "L_HIP": ffp(L_HIP),
            "R_HIP": ffp(R_HIP),
        }
        self.history.append(self.smooth_snapshot(raw))

    def detect(self):
        events = []
        hist = list(self.history)
        needed = PUNCH_LOOKBACK_FRAMES + PUNCH_CONFIRM_AFTER + 1
        if len(hist) < needed:
            return events
        peak_idx = len(hist) - 1 - PUNCH_CONFIRM_AFTER
        if peak_idx < PUNCH_LOOKBACK_FRAMES:
            return events
        peak_snap = hist[peak_idx]
        peak_frame = peak_snap["frame"]
        pre_snaps = hist[peak_idx - PUNCH_LOOKBACK_FRAMES:peak_idx]
        post_snaps = hist[peak_idx + 1:peak_idx + PUNCH_CONFIRM_AFTER + 1]
        local_snaps = pre_snaps + [peak_snap] + post_snaps

        for arm in ["left", "right"]:
            if peak_frame - self.last_event_frame[arm] < SAME_ARM_DEBOUNCE_FRAMES:
                continue
            peak_m = get_arm_metrics(peak_snap, arm)
            if peak_m is None:
                continue

            pre_m = [(s, get_arm_metrics(s, arm)) for s in pre_snaps]
            pre_m = [(s, m) for s, m in pre_m if m is not None]
            post_m = [(s, get_arm_metrics(s, arm)) for s in post_snaps]
            post_m = [(s, m) for s, m in post_m if m is not None]
            local_m = [(s, get_arm_metrics(s, arm)) for s in local_snaps]
            local_m = [(s, m) for s, m in local_m if m is not None]

            if len(pre_m) < 3 or len(post_m) < 1 or len(local_m) < 5:
                continue

            start_snap, start_m = min(pre_m, key=lambda x: x[1]["extension"])
            start_frame = start_snap["frame"]
            dur = peak_frame - start_frame
            if dur <= 0:
                continue

            max_local_ext = max(m["extension"] for _, m in local_m)
            if peak_m["extension"] < max_local_ext - 0.025:
                continue

            ext_delta = peak_m["extension"] - start_m["extension"]
            rel_wrist_move = peak_m["wrist_relative_to_shoulder"] - start_m["wrist_relative_to_shoulder"]
            rel_travel = float(np.linalg.norm(rel_wrist_move) / peak_m["torso_len"])
            abs_travel = dist_xy(peak_m["wrist"], start_m["wrist"]) / peak_m["torso_len"]
            rel_speed = rel_travel / max(1, dur)

            opp_center = peak_m["opponent_center"]
            opp_box = peak_m["opponent_box"]
            toward_score = None
            opp_dist_delta = None
            opp_gate = True

            if opp_center is not None:
                mu = unit_vec(rel_wrist_move)
                ou = unit_vec(xy(opp_center) - xy(start_m["shoulder"]))
                if mu is not None and ou is not None:
                    toward_score = float(np.dot(mu, ou))
                sd = float(np.linalg.norm(xy(start_m["wrist"]) - xy(opp_center)))
                pd = float(np.linalg.norm(xy(peak_m["wrist"]) - xy(opp_center)))
                opp_dist_delta = (sd - pd) / peak_m["torso_len"]
                wrist_near = point_in_expanded_box(peak_m["wrist"], opp_box, 0.30)
                opp_gate = wrist_near or (
                    (toward_score is None or toward_score >= -0.25)
                    and (opp_dist_delta is None or opp_dist_delta >= -0.20)
                )

            lh_peak = peak_snap["L_HIP"]
            rh_peak = peak_snap["R_HIP"]
            lh_start = start_snap["L_HIP"]
            rh_start = start_snap["R_HIP"]
            hip_angle_peak = math.degrees(math.atan2(rh_peak[1] - lh_peak[1], rh_peak[0] - lh_peak[0]))
            hip_angle_start = math.degrees(math.atan2(rh_start[1] - lh_start[1], rh_start[0] - lh_start[0]))
            hip_rotation_delta = angle_delta_deg(hip_angle_peak, hip_angle_start)

            is_punch = (
                opp_gate
                and 2 <= dur <= 12
                and ext_delta >= MIN_EXTENSION_DELTA
                and peak_m["extension"] >= MIN_PEAK_EXTENSION
                and rel_travel >= MIN_REL_WRIST_TRAVEL
                and abs_travel >= MIN_ABS_WRIST_TRAVEL
                and rel_speed >= MIN_REL_WRIST_SPEED
                and peak_m["elbow_angle"] >= MIN_ELBOW_ANGLE_AT_PEAK
            )

            if not is_punch:
                continue

            events.append({
                "fighter": int(peak_snap["fighter"]),
                "arm": arm,
                "peak_frame": int(peak_frame),
                "features": {
                    "duration_frames": int(dur),
                    "extension_start": float(start_m["extension"]),
                    "extension_peak": float(peak_m["extension"]),
                    "extension_delta": float(ext_delta),
                    "relative_wrist_travel": float(rel_travel),
                    "relative_wrist_speed": float(rel_speed),
                    "elbow_angle_start": float(start_m["elbow_angle"]),
                    "elbow_angle_peak": float(peak_m["elbow_angle"]),
                    "elbow_angle_delta": float(peak_m["elbow_angle"] - start_m["elbow_angle"]),
                    "toward_score": toward_score,
                    "opponent_distance_delta": opp_dist_delta,
                    "hip_rotation_delta": float(hip_rotation_delta),
                },
            })
            self.last_event_frame[arm] = peak_frame

        return events


class FighterTracker:
    def __init__(self, fighter_id):
        self.fighter_id = fighter_id
        self.pose = MP_POSE.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.signature = None
        self.punch_detector = SimplePunchDetector()

    def reset_punch_detector(self):
        self.punch_detector.reset()

    def update_signature(self, new_sig):
        if new_sig is None:
            return
        if self.signature is None:
            self.signature = new_sig
        else:
            self.signature = (1 - SIG_UPDATE_WEIGHT) * self.signature + SIG_UPDATE_WEIGHT * new_sig

    def process(self, img, box, frame_count, opponent_box=None, detect_punches=True):
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        ch, cw = crop.shape[:2]
        results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return []

        lm_list = results.pose_landmarks.landmark
        self.punch_detector.push_landmarks(frame_count, self.fighter_id, lm_list, box, opponent_box)
        punch_events = self.punch_detector.detect() if detect_punches else []

        for lid, lm in enumerate(lm_list):
            if lid in EXCLUDED_LANDMARKS:
                continue
            cx = int(lm.x * cw) + x1
            cy = int(lm.y * ch) + y1
            color = (255, 0, 0) if self.fighter_id == 0 else (0, 0, 255)
            cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        for conn in MP_POSE.POSE_CONNECTIONS:
            if conn[0] in EXCLUDED_LANDMARKS or conn[1] in EXCLUDED_LANDMARKS:
                continue
            s = lm_list[conn[0]]
            e = lm_list[conn[1]]
            sx, sy = int(s.x * cw) + x1, int(s.y * ch) + y1
            ex, ey = int(e.x * cw) + x1, int(e.y * ch) + y1
            color = (0, 255, 0) if self.fighter_id == 0 else (0, 165, 255)
            cv2.line(img, (sx, sy), (ex, ey), color, 2)

        return punch_events


def lock_to_biggest(detections, img):
    if len(detections) < 2:
        return None
    cands = sorted(detections.items(), key=lambda d: d[1]["fighter_score"], reverse=True)
    best_pair, best_score = None, -1.0
    for i, (ta, da) in enumerate(cands):
        ba = da["box"]
        for tb, db in cands[i + 1:]:
            bb = db["box"]
            overlap = iou(ba, bb)
            sep = abs(box_center_x(ba) - box_center_x(bb))
            if overlap >= 0.20 or sep < img.shape[1] * MIN_SEPARATION_RATIO:
                continue
            xa = box_center_x(ba) / img.shape[1]
            xb = box_center_x(bb) / img.shape[1]
            if (xa < 0.5 and xb < 0.5) or (xa > 0.5 and xb > 0.5):
                continue
            ps = (
                1200.0 * (da["fighter_score"] + db["fighter_score"])
                + box_area(ba)
                + box_area(bb)
                + sep
                + 1000.0 * (da["conf"] + db["conf"])
            )
            if ps > best_score:
                left, right = ((ta, ba), (tb, bb))
                if box_center_x(ba) > box_center_x(bb):
                    left, right = right, left
                best_pair = [left, right]
                best_score = ps
    if best_pair is None:
        return None
    return [(tid, box, appearance_signature(img, box)) for tid, box in best_pair]


def relock_by_appearance(trackers, detections, img):
    if not detections:
        return {}
    cands = []
    for fi, tracker in enumerate(trackers):
        if tracker.signature is None:
            continue
        for tid, det in detections.items():
            sig = appearance_signature(img, det["box"])
            score = hist_similarity(tracker.signature, sig) + 0.35 * det["fighter_score"]
            cands.append((score, fi, tid))
    cands.sort(reverse=True, key=lambda x: x[0])
    assigned_fighters, assigned_tracks, new_locks = set(), set(), {}
    for score, fi, tid in cands:
        if score < MIN_MATCH_SCORE:
            break
        if fi in assigned_fighters or tid in assigned_tracks:
            continue
        new_locks[fi] = tid
        assigned_fighters.add(fi)
        assigned_tracks.add(tid)
    return new_locks


def build_model_row(event):
    row = {"arm": event["arm"]}
    row.update(event["features"])
    return row


def classify_event(clf, event, confidence_threshold):
    row = build_model_row(event)
    X = pd.DataFrame([row])
    predicted_label = str(clf.predict(X)[0])
    confidence = None
    probabilities = {}

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        classes = [str(c) for c in clf.classes_]
        probabilities = {label: float(prob) for label, prob in zip(classes, proba)}
        confidence = float(max(proba))

    final_label = predicted_label
    if confidence is not None and confidence < confidence_threshold and "no_punch" not in probabilities:
        final_label = "likely_no_punch"

    return {
        "model_label": predicted_label,
        "final_label": final_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def reviewed_label(record):
    if record.get("punch") is False:
        return "no_punch"
    return record.get("punch_type", "unlabeled")


def load_actual_labels(labels_root, video_name):
    labels = []
    video_dir = labels_root / video_name
    if not video_dir.exists():
        return labels

    for json_path in sorted(video_dir.glob("punch_*/features.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        labels.append({
            "fighter": record.get("fighter"),
            "arm": record.get("arm"),
            "peak_frame": record.get("peak_frame"),
            "actual_label": reviewed_label(record),
            "actual_path": str(json_path),
        })
    return labels


def match_actual_label(event, actual_labels, frame_tolerance):
    matches = [
        label for label in actual_labels
        if label["fighter"] == event["fighter"] and label["arm"] == event["arm"]
    ]
    if not matches:
        return None

    closest = min(matches, key=lambda label: abs(label["peak_frame"] - event["peak_frame"]))
    frame_delta = abs(closest["peak_frame"] - event["peak_frame"])
    if frame_delta > frame_tolerance:
        return None

    return {
        "actual_label": closest["actual_label"],
        "actual_peak_frame": closest["peak_frame"],
        "actual_frame_delta": frame_delta,
        "actual_path": closest["actual_path"],
    }


def save_prediction_event(event, prediction, actual, raw_frame_buffer, clip_counter, video_fps, frame_w, frame_h, output_dir, save_clips):
    peak = event["peak_frame"]
    win_start = peak - CLIP_FRAMES_BEFORE
    win_end = peak + CLIP_FRAMES_AFTER
    folder = output_dir / f"event_{clip_counter:04d}_f{event['fighter']}_{event['arm']}_frame{peak}_{prediction['final_label']}"
    folder.mkdir(parents=True, exist_ok=True)

    if save_clips:
        frames = sorted(
            [(fc, fr) for fc, fr in raw_frame_buffer if win_start <= fc <= win_end],
            key=lambda x: x[0],
        )
        if frames:
            clip_path = folder / "clip.mp4"
            out = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (frame_w, frame_h))
            for _, frame in frames:
                out.write(frame)
            out.release()

    record = {
        "fighter": event["fighter"],
        "arm": event["arm"],
        "peak_frame": peak,
        "window_start": win_start,
        "window_end": win_end,
        "model_label": prediction["model_label"],
        "final_label": prediction["final_label"],
        "model_prediction": prediction["final_label"],
        "actual_label": actual["actual_label"] if actual else None,
        "actual_peak_frame": actual["actual_peak_frame"] if actual else None,
        "actual_frame_delta": actual["actual_frame_delta"] if actual else None,
        "actual_path": actual["actual_path"] if actual else None,
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "features": event["features"],
    }
    with open(folder / "prediction.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def draw_prediction_history(img, recent_predictions):
    y = 32
    for pred in list(recent_predictions)[-5:]:
        conf = pred["confidence"]
        conf_text = "n/a" if conf is None else f"{conf:.2f}"
        text = f"frame {pred['peak_frame']} F{pred['fighter']} {pred['arm']}: {pred['final_label']} ({conf_text})"
        cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        y += 28


def write_summary_csv(records, output_dir):
    if not records:
        return
    keys = [
        "peak_frame",
        "fighter",
        "arm",
        "model_prediction",
        "actual_label",
        "confidence",
        "model_label",
        "actual_peak_frame",
    ]
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in keys})


def run(video_path, model_path, yolo_path, output_root, labels_root, label_frame_tolerance, confidence_threshold, display, save_clips):
    project_dir = Path(__file__).resolve().parent
    video_path = Path(video_path)
    model_path = Path(model_path)
    yolo_path = Path(yolo_path)
    output_root = Path(output_root)
    labels_root = Path(labels_root)
    if not video_path.is_absolute():
        video_path = project_dir / video_path
    if not model_path.is_absolute():
        model_path = project_dir / model_path
    if not yolo_path.is_absolute():
        yolo_path = project_dir / yolo_path
    if not output_root.is_absolute():
        output_root = project_dir / output_root
    if not labels_root.is_absolute():
        labels_root = project_dir / labels_root

    tracker_path = project_dir / "custom_bytetrack.yaml"
    tracker_path.write_text(BYTETRACK_CONFIG, encoding="utf-8")

    clf = joblib.load(model_path)
    print(f"Loaded classifier: {model_path}")
    print(f"Model classes: {list(map(str, clf.classes_))}")
    if "no_punch" not in list(map(str, clf.classes_)):
        print(
            "Note: this model has no no_punch class. Low-confidence events will be labeled "
            f"likely_no_punch below {confidence_threshold:.2f} confidence."
        )

    yolo = YOLO(yolo_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = Path(video_path).stem
    output_dir = output_root / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_labels = load_actual_labels(labels_root, video_name)

    trackers = [FighterTracker(0), FighterTracker(1)]
    locked_ids = [None, None]
    prev_frame_sig = None
    missing_streak = 0
    fighter_missing_streaks = [0, 0]
    frame_count = 0
    ignore_punch_until_frame = -1
    raw_frame_buffer = deque(maxlen=RAW_FRAME_BUFFER_SIZE)
    pending_events = []
    records = []
    recent_predictions = deque(maxlen=5)
    event_counter = 1

    print(f"Running detection + classification on: {video_path}")
    print(f"Saving predictions to: {output_dir}")
    print(f"Loaded {len(actual_labels)} reviewed labels from: {labels_root / video_name}")

    while True:
        success, img = cap.read()
        if not success:
            break

        raw_frame_buffer.append((frame_count, img.copy()))
        h, w = img.shape[:2]

        cur_frame_sig = frame_signature(img)
        scene_cut = False
        if prev_frame_sig is not None and cv2.compareHist(prev_frame_sig, cur_frame_sig, cv2.HISTCMP_CORREL) < SCENE_CUT_THRESHOLD:
            scene_cut = True
        prev_frame_sig = cur_frame_sig

        if scene_cut:
            ignore_punch_until_frame = frame_count + SCENE_CUT_IGNORE_FRAMES
            for tracker in trackers:
                tracker.reset_punch_detector()

        results = yolo.track(
            img,
            classes=[0],
            conf=0.35,
            persist=True,
            verbose=False,
            tracker=str(tracker_path),
        )
        detections = {}
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box, track_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
                x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                cur_box = (x1, y1, x2, y2)
                if box_area(cur_box) < 0.04 * w * h:
                    continue
                score = fighter_candidate_score(img, cur_box, float(conf))
                if score < 0.12:
                    continue
                detections[int(track_id)] = {"box": cur_box, "conf": float(conf), "fighter_score": score}
        detections = dedupe_detections(detections)

        if all(lid is None for lid in locked_ids) and frame_count < INITIAL_LOCK_FRAMES:
            locks = lock_to_biggest(detections, img)
            if locks is not None:
                for i, (tid, _, sig) in enumerate(locks):
                    locked_ids[i] = tid
                    trackers[i].signature = sig
                print(f"[frame {frame_count}] Initial lock: {locked_ids}")

        for i, lid in enumerate(locked_ids):
            if lid is None:
                fighter_missing_streaks[i] = 0
                continue
            if lid in detections:
                fighter_missing_streaks[i] = 0
            else:
                fighter_missing_streaks[i] += 1
                if fighter_missing_streaks[i] >= SINGLE_FIGHTER_MISSING_UNLOCK:
                    locked_ids[i] = None
                    fighter_missing_streaks[i] = 0

        locked_present = [lid is not None for lid in locked_ids]
        visible = [lid is not None and lid in detections for lid in locked_ids]
        if any(locked_present) and not any(visible):
            missing_streak += 1
        else:
            missing_streak = 0

        need_relock = any(lid is None for lid in locked_ids) or scene_cut or missing_streak >= MISSING_FRAMES_RELOCK
        if need_relock and frame_count >= INITIAL_LOCK_FRAMES and len(detections) >= 1:
            new_locks = relock_by_appearance(trackers, detections, img)
            if new_locks:
                for fi, tid in new_locks.items():
                    if tid in locked_ids and locked_ids[fi] != tid:
                        continue
                    locked_ids[fi] = tid
                missing_streak = 0

        detect_punches = frame_count > ignore_punch_until_frame
        for i, fid in enumerate(locked_ids):
            if fid is None or fid not in detections:
                continue

            box = detections[fid]["box"]
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"Fighter {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            other_fid = locked_ids[1 - i]
            opp_box = detections[other_fid]["box"] if other_fid is not None and other_fid in detections else None
            punch_events = trackers[i].process(img, box, frame_count, opp_box, detect_punches)
            trackers[i].update_signature(appearance_signature(img, box))

            for event in punch_events:
                pending_events.append({"event": event, "save_at": event["peak_frame"] + CLIP_FRAMES_AFTER})

        still_pending = []
        for pending in pending_events:
            if frame_count < pending["save_at"]:
                still_pending.append(pending)
                continue
            prediction = classify_event(clf, pending["event"], confidence_threshold)
            actual = match_actual_label(pending["event"], actual_labels, label_frame_tolerance)
            record = save_prediction_event(
                pending["event"],
                prediction,
                actual,
                raw_frame_buffer,
                event_counter,
                video_fps,
                frame_w,
                frame_h,
                output_dir,
                save_clips,
            )
            records.append(record)
            recent_predictions.append(record)
            conf = record["confidence"]
            conf_text = "n/a" if conf is None else f"{conf:.3f}"
            print(
                f"[frame {record['peak_frame']}] fighter={record['fighter']} arm={record['arm']} "
                f"model_prediction={record['model_prediction']} actual={record['actual_label']} confidence={conf_text}"
            )
            event_counter += 1
        pending_events = still_pending

        if display:
            draw_prediction_history(img, recent_predictions)
            cv2.putText(img, f"F1={locked_ids[0]} F2={locked_ids[1]}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Punch Model Predictions", img)
            if cv2.waitKey(max(1, int(1000 / video_fps))) & 0xFF == ord("q"):
                break

        frame_count += 1

    for pending in pending_events:
        prediction = classify_event(clf, pending["event"], confidence_threshold)
        actual = match_actual_label(pending["event"], actual_labels, label_frame_tolerance)
        record = save_prediction_event(
            pending["event"],
            prediction,
            actual,
            raw_frame_buffer,
            event_counter,
            video_fps,
            frame_w,
            frame_h,
            output_dir,
            save_clips,
        )
        records.append(record)
        event_counter += 1

    write_summary_csv(records, output_dir)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Classified {len(records)} candidate events.")
    print(f"Summary: {output_dir / 'summary.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run merge.py-style punch detection on a video and classify each candidate with the saved model."
    )
    parser.add_argument("video", help="Video path to test, for example video_data/test_video5.mp4")
    parser.add_argument("--model", default=MODEL_PATH, help="Saved classifier path.")
    parser.add_argument("--yolo", default=YOLO_MODEL_PATH, help="YOLO model path.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Directory for prediction JSON/CSV output.")
    parser.add_argument("--labels-root", default="data_collection", help="Root directory containing reviewed labels.")
    parser.add_argument(
        "--label-frame-tolerance",
        type=int,
        default=3,
        help="Maximum frame difference allowed when matching a prediction to a reviewed label.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.45,
        help="If the model has no no_punch class, predictions below this confidence become likely_no_punch.",
    )
    parser.add_argument("--no-display", action="store_true", help="Run without opening the OpenCV preview window.")
    parser.add_argument("--no-clips", action="store_true", help="Save prediction JSON only, without event clips.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        video_path=args.video,
        model_path=args.model,
        yolo_path=args.yolo,
        output_root=args.output,
        labels_root=args.labels_root,
        label_frame_tolerance=args.label_frame_tolerance,
        confidence_threshold=args.confidence_threshold,
        display=not args.no_display,
        save_clips=not args.no_clips,
    )
