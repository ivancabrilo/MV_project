
import cv2
import mediapipe as mp
import json
import os
import math
import shutil
import numpy as np
from collections import deque
from ultralytics import YOLO

VIDEO_PATH = "test_video2.mp4"
MODEL_PATH = "yolo26n.pt"
DATA_COLLECTION_DIR = "data_collection"

# Set to True to wipe all data_collection contents before each run.
# Set to False to keep accumulating data across runs.
CLEAR_DATA_ON_RUN = False

# ============================================================
# TRACKING / RE-LOCKING TUNING
# ============================================================

SCENE_CUT_THRESHOLD = 0.6
MISSING_FRAMES_RELOCK = 10
SINGLE_FIGHTER_MISSING_UNLOCK = 6
MIN_MATCH_SCORE = 0.5
SIG_UPDATE_WEIGHT = 0.1
INITIAL_LOCK_FRAMES = 20
MIN_SEPARATION_RATIO = 0.18

# ============================================================
# PUNCH DETECTION TUNING
# ============================================================

SCENE_CUT_IGNORE_FRAMES = 8
CLIP_FRAMES_BEFORE = 14
CLIP_FRAMES_AFTER = 12
PUNCH_LOOKBACK_FRAMES = 7
PUNCH_CONFIRM_AFTER = 2
MIN_ARM_VISIBILITY = 0.35 # was 0.45
MIN_EXTENSION_DELTA = 0.12
MIN_PEAK_EXTENSION = 0.90
MIN_REL_WRIST_TRAVEL = 0.14
MIN_ABS_WRIST_TRAVEL = 0.16
MIN_REL_WRIST_SPEED = 0.020
MIN_ELBOW_ANGLE_AT_PEAK = 70
SAME_ARM_DEBOUNCE_FRAMES = 5
SMOOTH_ALPHA = 0.50 #0.6
RAW_FRAME_BUFFER_SIZE = 120
LANDMARK_HISTORY_SIZE = 160

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

ARM_KEYS = {
    "left": {"shoulder": "L_SHOULDER", "elbow": "L_ELBOW", "wrist": "L_WRIST", "hip": "L_HIP"},
    "right": {"shoulder": "R_SHOULDER", "elbow": "R_ELBOW", "wrist": "R_WRIST", "hip": "R_HIP"},
}

excluded_landmarks = list(range(0, 11)) + list(range(25, 33))
mpPose = mp.solutions.pose


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_center_x(box):
    x1, _, x2, _ = box
    return (x1 + x2) / 2.0

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

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
    for tid, det in sorted(detections.items(),
                           key=lambda item: (item[1]["conf"], box_area(item[1]["box"])),
                           reverse=True):
        box = det["box"]
        is_dup = False
        for kept_det in kept.values():
            kb = kept_det["box"]
            if (iou(box, kb) >= overlap_thresh or contains_center(box, kb)
                    or contains_center(kb, box) or center_distance(box, kb) <= center_dist_thresh):
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
    return (x1 - bw * expand_ratio <= point[0] <= x2 + bw * expand_ratio
            and y1 - bh * expand_ratio <= point[1] <= y2 + bh * expand_ratio)

def distance_to_box(point, box):
    if box is None:
        return None
    px, py = point[0], point[1]
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return float((dx ** 2 + dy ** 2) ** 0.5)


# ============================================================
# APPEARANCE / RE-LOCKING HELPERS
# ============================================================

def torso_crop(img, box):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1 = x1 + int(bw * 0.2); tx2 = x2 - int(bw * 0.2)
    ty1 = y1 + int(bh * 0.25); ty2 = y2 - int(bh * 0.35)
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
    center_x_norm = cx / w; center_y_norm = cy / h
    bottom_norm = box[3] / h; top_norm = box[1] / h
    height_norm = (box[3] - box[1]) / h
    return (2.2 * area_norm + 0.9 * bottom_norm + 0.6 * center_y_norm + 0.5 * conf
            - abs(center_x_norm - 0.5) * 1.2
            - dark_torso_ratio(img, box) * 0.45
            - max(0.0, 0.42 - center_y_norm) * 3.5
            - max(0.0, 0.18 - top_norm) * 2.5
            - max(0.0, height_norm - 0.62) * 2.5)


# ============================================================
# PUNCH METRIC HELPER
# ============================================================

def get_arm_metrics(snapshot, arm):
    # distinguishing arms
    keys = ARM_KEYS[arm]
    shoulder = snapshot[keys["shoulder"]]
    elbow = snapshot[keys["elbow"]]
    wrist = snapshot[keys["wrist"]]
    hip = snapshot[keys["hip"]]
    vis_min = min(shoulder[3], elbow[3], wrist[3], hip[3])
    if vis_min < MIN_ARM_VISIBILITY: # might want to tweak this
        return None
    torso_len = dist_xy(shoulder, hip)
    if torso_len < 1e-6:
        return None
    return {
        "shoulder": shoulder, "elbow": elbow, "wrist": wrist, "hip": hip,
        "visibility_min": float(vis_min),
        "torso_len": float(torso_len),
        "extension": dist_xy(shoulder, wrist) / torso_len,
        "elbow_angle": angle_3pt(shoulder, elbow, wrist),
        "wrist_relative_to_shoulder": xy(wrist) - xy(shoulder),
        "opponent_center": snapshot.get("opponent_center"),
        "opponent_box": snapshot.get("opponent_box"),
    }


# ============================================================
# PUNCH DETECTOR (unchanged logic)
# ============================================================

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
        smoothed = {"frame": raw["frame"], "fighter": raw["fighter"],
                    "fighter_box": raw["fighter_box"], "opponent_box": raw["opponent_box"],
                    "opponent_center": raw["opponent_center"]}
        for key in ["L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW",
                     "L_WRIST", "R_WRIST", "L_HIP", "R_HIP"]:
            cur, prev = raw[key], self.prev_smoothed[key]
            smoothed[key] = (float((1-SMOOTH_ALPHA)*prev[0] + SMOOTH_ALPHA*cur[0]),
                             float((1-SMOOTH_ALPHA)*prev[1] + SMOOTH_ALPHA*cur[1]),
                             float((1-SMOOTH_ALPHA)*prev[2] + SMOOTH_ALPHA*cur[2]),
                             float(cur[3]))
        self.prev_smoothed = smoothed
        return smoothed

    def push_landmarks(self, frame_count, fighter_id, landmarks, fighter_box, opponent_box=None):
        x1, y1, x2, y2 = fighter_box
        bw, bh = x2 - x1, y2 - y1
        opp_center = box_center(opponent_box) if opponent_box is not None else None

        def ffp(idx):
            lm = landmarks[idx]
            return (float(lm.x * bw + x1), float(lm.y * bh + y1),
                    float(lm.z * bw), float(lm.visibility))

        raw = {
            "frame": int(frame_count), "fighter": int(fighter_id),
            "fighter_box": tuple(map(float, fighter_box)),
            "opponent_box": tuple(map(float, opponent_box)) if opponent_box else None,
            "opponent_center": tuple(map(float, opp_center)) if opp_center else None,
            "L_SHOULDER": ffp(L_SHOULDER), "R_SHOULDER": ffp(R_SHOULDER),
            "L_ELBOW": ffp(L_ELBOW), "R_ELBOW": ffp(R_ELBOW),
            "L_WRIST": ffp(L_WRIST), "R_WRIST": ffp(R_WRIST),
            "L_HIP": ffp(L_HIP), "R_HIP": ffp(R_HIP),
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

            # Opponent gate
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
                opp_gate = (wrist_near
                            or ((toward_score is None or toward_score >= -0.25)
                                and (opp_dist_delta is None or opp_dist_delta >= -0.20)))

            # Hip rotation (added for Random Forest — cross vs jab)
            lh_peak = peak_snap["L_HIP"]; rh_peak = peak_snap["R_HIP"]
            lh_start = start_snap["L_HIP"]; rh_start = start_snap["R_HIP"]
            hip_angle_peak = math.degrees(math.atan2(rh_peak[1]-lh_peak[1], rh_peak[0]-lh_peak[0]))
            hip_angle_start = math.degrees(math.atan2(rh_start[1]-lh_start[1], rh_start[0]-lh_start[0]))
            hip_rotation_delta = hip_angle_peak - hip_angle_start

            is_punch = (opp_gate
                        and 2 <= dur <= 12
                        and ext_delta >= MIN_EXTENSION_DELTA
                        and peak_m["extension"] >= MIN_PEAK_EXTENSION
                        and rel_travel >= MIN_REL_WRIST_TRAVEL
                        and abs_travel >= MIN_ABS_WRIST_TRAVEL
                        and rel_speed >= MIN_REL_WRIST_SPEED
                        and peak_m["elbow_angle"] >= MIN_ELBOW_ANGLE_AT_PEAK)

            if not is_punch:
                continue

            # Only keep features the Random Forest needs for punch-type classification
            features = {
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
            }

            events.append({
                "fighter": int(peak_snap["fighter"]),
                "arm": arm,
                "peak_frame": int(peak_frame),
                "features": features,
            })
            self.last_event_frame[arm] = peak_frame

        return events


# ============================================================
# FIGHTER TRACKER
# ============================================================

class FighterTracker:
    def __init__(self, fighter_id):
        self.fighter_id = fighter_id
        self.pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.data = []
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

        frame_data = {"frame": frame_count, "fighter": self.fighter_id, "landmarks": []}
        for lid, lm in enumerate(lm_list):
            if lid not in excluded_landmarks:
                cx = int(lm.x * cw) + x1
                cy = int(lm.y * ch) + y1
                color = (255, 0, 0) if self.fighter_id == 0 else (0, 0, 255)
                cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)
                if lm.visibility > 0.35:
                    frame_data["landmarks"].append({"id": lid, "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})

        for conn in mpPose.POSE_CONNECTIONS:
            if conn[0] not in excluded_landmarks and conn[1] not in excluded_landmarks:
                s = lm_list[conn[0]]; e = lm_list[conn[1]]
                sx, sy = int(s.x * cw) + x1, int(s.y * ch) + y1
                ex, ey = int(e.x * cw) + x1, int(e.y * ch) + y1
                color = (0, 255, 0) if self.fighter_id == 0 else (0, 165, 255)
                cv2.line(img, (sx, sy), (ex, ey), color, 2)

        for ev in punch_events:
            wi = L_WRIST if ev["arm"] == "left" else R_WRIST
            wlm = lm_list[wi]
            wx, wy = int(wlm.x * cw) + x1, int(wlm.y * ch) + y1
            cv2.circle(img, (wx, wy), 18, (0, 255, 255), 3)
            cv2.putText(img, f"PUNCH {ev['arm'][0].upper()}", (wx+18, wy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        self.data.append(frame_data)
        return punch_events


# ============================================================
# LOCKING / RE-LOCKING (unchanged)
# ============================================================

def lock_to_biggest(detections, img):
    if len(detections) < 2:
        return None
    cands = sorted(detections.items(), key=lambda d: d[1]["fighter_score"], reverse=True)
    best_pair, best_score = None, -1.0
    for i, (ta, da) in enumerate(cands):
        ba = da["box"]
        for tb, db in cands[i+1:]:
            bb = db["box"]
            overlap = iou(ba, bb)
            sep = abs(box_center_x(ba) - box_center_x(bb))
            if overlap >= 0.20 or sep < img.shape[1] * MIN_SEPARATION_RATIO:
                continue
            xa = box_center_x(ba) / img.shape[1]; xb = box_center_x(bb) / img.shape[1]
            if (xa < 0.5 and xb < 0.5) or (xa > 0.5 and xb > 0.5):
                continue
            ps = (1200.0 * (da["fighter_score"] + db["fighter_score"])
                  + box_area(ba) + box_area(bb) + sep
                  + 1000.0 * (da["conf"] + db["conf"]))
            if ps > best_score:
                left, right = ((ta, ba), (tb, bb))
                if box_center_x(ba) > box_center_x(bb):
                    left, right = right, left
                best_pair = [left, right]; best_score = ps
    if best_pair is None:
        return None
    return [(tid, box, appearance_signature(img, box)) for tid, box in best_pair]


def relock_by_appearance(trackers, detections, img):
    if not detections:
        return {}
    cands = []
    for fi, t in enumerate(trackers):
        if t.signature is None:
            continue
        for tid, det in detections.items():
            sig = appearance_signature(img, det["box"])
            score = hist_similarity(t.signature, sig) + 0.35 * det["fighter_score"]
            cands.append((score, fi, tid))
    cands.sort(reverse=True, key=lambda x: x[0])
    af, at, nl = set(), set(), {}
    for score, fi, tid in cands:
        if score < MIN_MATCH_SCORE:
            break
        if fi in af or tid in at:
            continue
        nl[fi] = tid; af.add(fi); at.add(tid)
    return nl


# ============================================================
# SAVING — trimmed JSON, per-video folder
# ============================================================

def save_punch_event(event, raw_frame_buffer, clip_counter, video_fps, frame_w, frame_h, video_output_dir):
    peak = event["peak_frame"]
    win_start = peak - CLIP_FRAMES_BEFORE
    win_end = peak + CLIP_FRAMES_AFTER
    frames = sorted([(fc, fr) for fc, fr in raw_frame_buffer if win_start <= fc <= win_end],
                    key=lambda x: x[0])
    if len(frames) < 5:
        return clip_counter

    folder = f"punch_{clip_counter:04d}_f{event['fighter']}_{event['arm']}_frame{peak}"
    folder_path = os.path.join(video_output_dir, folder)
    os.makedirs(folder_path, exist_ok=True)

    clip_path = os.path.join(folder_path, "clip.mp4")
    out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (frame_w, frame_h))
    for _, fr in frames:
        out.write(fr)
    out.release()

    record = {
        "fighter": event["fighter"],
        "arm": event["arm"],
        "peak_frame": peak,
        "window_start": win_start,
        "window_end": win_end,
        "punch": True,
        "punch_type": "unlabeled",
        "features": event["features"],
    }
    with open(os.path.join(folder_path, "features.json"), "w") as f:
        json.dump(record, f, indent=2)

    print(f"  -> SAVED {folder}")
    return clip_counter + 1


# ============================================================
# BYTETRACK CONFIG
# ============================================================

BYTETRACK_CONFIG = """
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.5
track_buffer: 60
match_thresh: 0.7
fuse_score: True
"""
with open("custom_bytetrack.yaml", "w") as f:
    f.write(BYTETRACK_CONFIG)


# ============================================================
# MAIN
# ============================================================

# Clear data if flag is set
if CLEAR_DATA_ON_RUN and os.path.exists(DATA_COLLECTION_DIR):
    shutil.rmtree(DATA_COLLECTION_DIR)
    print(f"Cleared {DATA_COLLECTION_DIR}/")

os.makedirs(DATA_COLLECTION_DIR, exist_ok=True)

# Per-video subfolder
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
video_output_dir = os.path.join(DATA_COLLECTION_DIR, video_name)
os.makedirs(video_output_dir, exist_ok=True)
print(f"Saving punches to: {video_output_dir}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

model = YOLO(MODEL_PATH)
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

trackers = [FighterTracker(0), FighterTracker(1)]
locked_ids = [None, None]
prev_frame_sig = None
missing_streak = 0
fighter_missing_streaks = [0, 0]
frame_count = 0
ignore_punch_until_frame = -1
raw_frame_buffer = deque(maxlen=RAW_FRAME_BUFFER_SIZE)
pending_events = []
clip_counter = 1

while True:
    success, img = cap.read()
    if not success:
        break

    raw_frame_buffer.append((frame_count, img.copy()))
    h, w = img.shape[:2]

    # Scene cut
    cur_frame_sig = frame_signature(img)
    scene_cut = False
    if prev_frame_sig is not None:
        if cv2.compareHist(prev_frame_sig, cur_frame_sig, cv2.HISTCMP_CORREL) < SCENE_CUT_THRESHOLD:
            scene_cut = True
    prev_frame_sig = cur_frame_sig

    if scene_cut:
        ignore_punch_until_frame = frame_count + SCENE_CUT_IGNORE_FRAMES
        for t in trackers:
            t.reset_punch_detector()

    # YOLO + ByteTrack
    results = model.track(img, classes=[0], conf=0.35, persist=True, verbose=False,
                          tracker="custom_bytetrack.yaml")
    detections = {}
    if results[0].boxes is not None and results[0].boxes.id is not None:
        for box, track_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
            x2, y2 = max(0, min(x2, w-1)), max(0, min(y2, h-1))
            if x2 <= x1 or y2 <= y1:
                continue
            tid = int(track_id)
            cur_box = (x1, y1, x2, y2)
            if box_area(cur_box) < 0.04 * w * h:
                continue
            score = fighter_candidate_score(img, cur_box, float(conf))
            if score < 0.12:
                continue
            detections[tid] = {"box": cur_box, "conf": float(conf), "fighter_score": score}
    detections = dedupe_detections(detections)

    # Initial lock
    if all(lid is None for lid in locked_ids) and frame_count < INITIAL_LOCK_FRAMES:
        locks = lock_to_biggest(detections, img)
        if locks is not None:
            for i, (tid, box, sig) in enumerate(locks):
                locked_ids[i] = tid; trackers[i].signature = sig
            print(f"[frame {frame_count}] Initial lock: {locked_ids}")

    # Missing / unlock
    for i, lid in enumerate(locked_ids):
        if lid is None:
            fighter_missing_streaks[i] = 0; continue
        if lid in detections:
            fighter_missing_streaks[i] = 0
        else:
            fighter_missing_streaks[i] += 1
            if fighter_missing_streaks[i] >= SINGLE_FIGHTER_MISSING_UNLOCK:
                print(f"[frame {frame_count}] Unlock fighter {i}: lost track {lid}")
                locked_ids[i] = None; fighter_missing_streaks[i] = 0

    locked_present = [lid is not None for lid in locked_ids]
    visible = [lid is not None and lid in detections for lid in locked_ids]
    if any(locked_present) and not any(visible):
        missing_streak += 1
    else:
        missing_streak = 0

    need_relock = (any(lid is None for lid in locked_ids) or scene_cut
                   or missing_streak >= MISSING_FRAMES_RELOCK)

    if need_relock and frame_count >= INITIAL_LOCK_FRAMES and len(detections) >= 1:
        new_locks = relock_by_appearance(trackers, detections, img)
        if new_locks:
            for fi, tid in new_locks.items():
                if tid in locked_ids and locked_ids[fi] != tid:
                    continue
                if locked_ids[fi] != tid:
                    reason = 'scene cut' if scene_cut else ('missing' if missing_streak else 'unlocked')
                    print(f"[frame {frame_count}] Re-lock fighter {fi}: {locked_ids[fi]} -> {tid} ({reason})")
                locked_ids[fi] = tid
            missing_streak = 0

    # Process fighters
    detect_punches = frame_count > ignore_punch_until_frame
    for i, fid in enumerate(locked_ids):
        if fid is None or fid not in detections:
            continue
        box = detections[fid]["box"]
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"Fighter {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        opp_box = None
        other_fid = locked_ids[1 - i]
        if other_fid is not None and other_fid in detections:
            opp_box = detections[other_fid]["box"]

        punch_events = trackers[i].process(img, box, frame_count, opp_box, detect_punches)
        trackers[i].update_signature(appearance_signature(img, box))

        for ev in punch_events:
            pending_events.append({"event": ev, "tracker_index": i,
                                   "save_at": ev["peak_frame"] + CLIP_FRAMES_AFTER})

    # Save pending
    still_pending = []
    for p in pending_events:
        if frame_count < p["save_at"]:
            still_pending.append(p); continue
        clip_counter = save_punch_event(p["event"], raw_frame_buffer, clip_counter,
                                        video_fps, frame_w, frame_h, video_output_dir)
    pending_events = still_pending

    # HUD
    if scene_cut:
        cv2.putText(img, "SCENE CUT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    status = f"F1={locked_ids[0]} F2={locked_ids[1]} pending={len(pending_events)}"
    cv2.putText(img, status, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Fighters", img)
    if cv2.waitKey(max(1, int(1000 / video_fps))) & 0xFF == ord('q'):
        break
    frame_count += 1

# End-of-video flush
for p in pending_events:
    clip_counter = save_punch_event(p["event"], raw_frame_buffer, clip_counter,
                                    video_fps, frame_w, frame_h, video_output_dir)

for tracker in trackers:
    with open(f'landmarks_fighter_{tracker.fighter_id}.json', 'w') as f:
        json.dump(tracker.data, f, indent=4)
    print(f"Fighter {tracker.fighter_id}: {len(tracker.data)} frames captured")

cap.release()
cv2.destroyAllWindows()