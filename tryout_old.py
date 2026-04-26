import cv2
import mediapipe as mp
import json
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "test_video4.mp4"
MODEL_PATH = "yolo26n.pt"

# --- Re-locking tuning knobs ---
SCENE_CUT_THRESHOLD = 0.6      # histogram correlation below this = camera cut (camera view changed suddenly)
MISSING_FRAMES_RELOCK = 10     # both fighters missing this long = trigger re-lock (fallback)
MIN_MATCH_SCORE = 0.5          # min histogram correlation to accept a re-lock match (only accept a re-lock if the match score is at least 0.5)
SIG_UPDATE_WEIGHT = 0.1        # this parameter controls how fast that signature changes (refreshing stored signatures over time)
INITIAL_LOCK_FRAMES = 20       # frames to wait before locking initially

excluded_landmarks = list(range(0, 11)) + list(range(25, 33))
mpPose = mp.solutions.pose


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def torso_crop(img, box): # can anything else be done here to fix locking on correct fighter 
    """Take the middle 60% of the bounding box (mostly torso and shorts, avoids head/legs/background)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1 = x1 + int(bw * 0.2)
    tx2 = x2 - int(bw * 0.2)
    ty1 = y1 + int(bh * 0.30) # (maybe we need to do more) it was0.25 
    ty2 = y2 - int(bh * 0.20) # (maybe we need to do more) it was0.35 
    # have in mind that fighters are different height - but do boxes adjust for that?
    crop = img[max(0, ty1):ty2, max(0, tx1):tx2]
    return crop

# building histogram for each fighter - hopefully calculates color of shorts for the hist value
def appearance_signature(img, box):
    """HSV color histogram of the torso region. Returns None if crop is empty."""
    crop = torso_crop(img, box)
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 2D hist over hue + saturation (ignore value to be lighting-robust)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist 


def hist_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

# used to detect camera cuts/angle switch
def frame_signature(img):
    """Coarse grayscale histogram of the whole frame, for scene-cut detection."""
    small = cv2.resize(img, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


# GO THROUGH THIS
class FighterTracker:
    def __init__(self, fighter_id):
        self.fighter_id = fighter_id
        self.pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.data = []
        self.signature = None  # HSV histogram, set when first locked

    def update_signature(self, new_sig):
        if new_sig is None:
            return
        if self.signature is None:
            self.signature = new_sig
        else:
            # Exponential moving average to slowly adapt to lighting/angle changes
            self.signature = (1 - SIG_UPDATE_WEIGHT) * self.signature + SIG_UPDATE_WEIGHT * new_sig

    def process(self, img, box, frame_count):
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return
        ch, cw = crop.shape[:2]

        results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) # results is MediaPipe object (self.pose = mpPose.Pose...)
        if not results.pose_landmarks:
            return

        frame_data = {"frame": frame_count, "fighter": self.fighter_id, "landmarks": []}
        # id is index for body part
        for id, lm in enumerate(results.pose_landmarks.landmark): # lm is landmark object (x,y,z, visibility)
            if id not in excluded_landmarks:
                cx = int(lm.x * cw) + x1
                cy = int(lm.y * ch) + y1
                color = (255, 0, 0) if self.fighter_id == 0 else (0, 0, 255)
                cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

                if lm.visibility > 0.35:
                    frame_data["landmarks"].append({
                        "id": id,
                        "x": lm.x, "y": lm.y, "z": lm.z,
                        "visibility": lm.visibility
                    })

        for connection in mpPose.POSE_CONNECTIONS:
            if connection[0] not in excluded_landmarks and connection[1] not in excluded_landmarks:
                start = results.pose_landmarks.landmark[connection[0]]
                end = results.pose_landmarks.landmark[connection[1]]
                sx = int(start.x * cw) + x1
                sy = int(start.y * ch) + y1
                ex = int(end.x * cw) + x1
                ey = int(end.y * ch) + y1
                color = (0, 255, 0) if self.fighter_id == 0 else (0, 165, 255)
                cv2.line(img, (sx, sy), (ex, ey), color, 2)

        self.data.append(frame_data)


def lock_to_biggest(detections, img):
    """Initial lock: pick 2 biggest detections, return [(tid, box, signature), ...]."""
    if len(detections) < 2:
        return None
    top2 = sorted(detections.items(), key=lambda d: box_area(d[1]), reverse=True)[:2]
    result = []
    for tid, box in top2:
        sig = appearance_signature(img, box)
        result.append((tid, box, sig))
    return result


def relock_by_appearance(trackers, detections, img):
    """
    Greedy match each existing tracker to its best-scoring detection by histogram correlation.
    Returns a dict {fighter_index: track_id} for fighters that found a confident match.
    """
    if not detections:
        return {}

    # Score every (fighter, detection) pair
    candidates = []  # (score, fighter_idx, tid)
    for fi, tracker in enumerate(trackers):
        if tracker.signature is None:
            continue
        for tid, box in detections.items():
            sig = appearance_signature(img, box)
            score = hist_similarity(tracker.signature, sig)
            candidates.append((score, fi, tid))

    candidates.sort(reverse=True, key=lambda x: x[0])  # best score first

    assigned_fighters = set()
    assigned_tids = set()
    new_locks = {}
    for score, fi, tid in candidates:
        if score < MIN_MATCH_SCORE:
            break  # remaining are even worse
        if fi in assigned_fighters or tid in assigned_tids:
            continue
        new_locks[fi] = tid
        assigned_fighters.add(fi)
        assigned_tids.add(tid)

    return new_locks


# --- ByteTrack config ---
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


cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(MODEL_PATH)
video_fps = cap.get(cv2.CAP_PROP_FPS)

trackers = [FighterTracker(0), FighterTracker(1)]
locked_ids = [None, None]   # parallel to trackers; None means that fighter is currently unlocked
prev_frame_sig = None
missing_streak = 0
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    h, w = img.shape[:2]

    # --- Scene cut detection ---
    cur_frame_sig = frame_signature(img)
    scene_cut = False
    if prev_frame_sig is not None:
        sim = cv2.compareHist(prev_frame_sig, cur_frame_sig, cv2.HISTCMP_CORREL)
        if sim < SCENE_CUT_THRESHOLD:
            scene_cut = True
    prev_frame_sig = cur_frame_sig

    # --- YOLO + ByteTrack ---
    results = model.track(
        img,
        classes=[0],
        conf=0.35,
        persist=True,
        verbose=False,
        tracker="custom_bytetrack.yaml"
    )

    detections = {}
    if results[0].boxes is not None and results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box)

            pad_x = int((x2 - x1) * 0.3)   # 30% width padding
            pad_y = int((y2 - y1) * 0.1)   # 10% height padding

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            tid = int(track_id)
            if box_area((x1, y1, x2, y2)) < 0.04 * w * h:
                continue
            detections[tid] = (x1, y1, x2, y2)

    # --- Initial lock ---
    if all(lid is None for lid in locked_ids) and frame_count < INITIAL_LOCK_FRAMES:
        locks = lock_to_biggest(detections, img)
        if locks is not None:
            for i, (tid, box, sig) in enumerate(locks):
                locked_ids[i] = tid
                trackers[i].signature = sig
            print(f"[frame {frame_count}] Initial lock: {locked_ids}")

    # --- Decide whether to re-lock ---
    locked_present = [lid is not None for lid in locked_ids]
    visible = [lid is not None and lid in detections for lid in locked_ids]
    if any(locked_present) and not any(visible):
        missing_streak += 1
    else:
        missing_streak = 0

    need_relock = (
        any(lid is None for lid in locked_ids) or       # at least one fighter unlocked
        scene_cut or                                     # camera cut
        missing_streak >= MISSING_FRAMES_RELOCK          # both gone for too long
    )

    if need_relock and frame_count >= INITIAL_LOCK_FRAMES and len(detections) >= 1:
        new_locks = relock_by_appearance(trackers, detections, img)
        if new_locks:
            for fi, tid in new_locks.items():
                if locked_ids[fi] != tid:
                    reason = 'scene cut' if scene_cut else ('missing' if missing_streak else 'unlocked')
                    print(f"[frame {frame_count}] Re-lock fighter {fi}: {locked_ids[fi]} -> {tid} ({reason})")
                locked_ids[fi] = tid
            missing_streak = 0

    # --- Process locked fighters ---
    for i, fid in enumerate(locked_ids):
        if fid is None or fid not in detections:
            continue
        box = detections[fid]
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"Fighter {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        trackers[i].process(img, box, frame_count)
        # Slowly refresh appearance signature with current crop
        trackers[i].update_signature(appearance_signature(img, box))

    # --- HUD ---
    if scene_cut:
        cv2.putText(img, "SCENE CUT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    status = f"F1={locked_ids[0]} F2={locked_ids[1]}"
    cv2.putText(img, status, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Fighters", img)
    if cv2.waitKey(max(1, int(1000 / video_fps))) & 0xFF == ord('q'):
        break

    frame_count += 1

for tracker in trackers:
    with open(f'landmarks_fighter_{tracker.fighter_id}.json', 'w') as f:
        json.dump(tracker.data, f, indent=4)
    print(f"Fighter {tracker.fighter_id}: {len(tracker.data)} frames captured")

cap.release()
cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import json
# import os
# import math
# import numpy as np
# from collections import deque
# from ultralytics import YOLO

# VIDEO_PATH = "test_video4.mp4"
# MODEL_PATH = "yolo26n.pt"
# DATA_COLLECTION_DIR = "data_collection"

# # ============================================================
# # ORIGINAL TRACKING / RE-LOCKING TUNING
# # ============================================================

# SCENE_CUT_THRESHOLD = 0.6
# MISSING_FRAMES_RELOCK = 10
# SINGLE_FIGHTER_MISSING_UNLOCK = 6
# MIN_MATCH_SCORE = 0.5
# SIG_UPDATE_WEIGHT = 0.1
# INITIAL_LOCK_FRAMES = 20
# MIN_SEPARATION_RATIO = 0.18

# # ============================================================
# # SIMPLE PUNCH DETECTION TUNING
# # ============================================================

# # After a scene cut, skip punch detection for a few frames.
# # Tracking/pose may be unstable immediately after cuts.
# SCENE_CUT_IGNORE_FRAMES = 8

# # Clip length around detected punch peak.
# CLIP_FRAMES_BEFORE = 14
# CLIP_FRAMES_AFTER = 12

# # Punch detection uses a small temporal window.
# # Candidate peak is confirmed after a couple of future frames.
# PUNCH_LOOKBACK_FRAMES = 7
# PUNCH_CONFIRM_AFTER = 2

# # Minimum arm landmark visibility.
# MIN_ARM_VISIBILITY = 0.45

# # Main punch thresholds.
# # These are intentionally simple and based on full-frame coordinates.
# MIN_EXTENSION_DELTA = 0.12
# MIN_PEAK_EXTENSION = 0.90
# MIN_REL_WRIST_TRAVEL = 0.14
# MIN_ABS_WRIST_TRAVEL = 0.16
# MIN_REL_WRIST_SPEED = 0.020
# MIN_ELBOW_ANGLE_AT_PEAK = 70

# # A very small debounce prevents duplicate detections from the same motion,
# # but still allows fast combos.
# SAME_ARM_DEBOUNCE_FRAMES = 5

# # Lightweight smoothing. Higher = follows movement faster.
# SMOOTH_ALPHA = 0.60

# RAW_FRAME_BUFFER_SIZE = 120
# LANDMARK_HISTORY_SIZE = 160

# # ============================================================
# # MEDIAPIPE SETUP
# # ============================================================

# L_SHOULDER, R_SHOULDER = 11, 12
# L_ELBOW, R_ELBOW = 13, 14
# L_WRIST, R_WRIST = 15, 16
# L_HIP, R_HIP = 23, 24

# ARM_KEYS = {
#     "left": {
#         "shoulder": "L_SHOULDER",
#         "elbow": "L_ELBOW",
#         "wrist": "L_WRIST",
#         "hip": "L_HIP",
#     },
#     "right": {
#         "shoulder": "R_SHOULDER",
#         "elbow": "R_ELBOW",
#         "wrist": "R_WRIST",
#         "hip": "R_HIP",
#     },
# }

# excluded_landmarks = list(range(0, 11)) + list(range(25, 33))
# mpPose = mp.solutions.pose


# # ============================================================
# # BASIC GEOMETRY HELPERS
# # ============================================================

# def box_area(box):
#     x1, y1, x2, y2 = box
#     return max(0, x2 - x1) * max(0, y2 - y1)


# def box_center_x(box):
#     x1, _, x2, _ = box
#     return (x1 + x2) / 2.0


# def box_center(box):
#     x1, y1, x2, y2 = box
#     return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# def iou(box_a, box_b):
#     ax1, ay1, ax2, ay2 = box_a
#     bx1, by1, bx2, by2 = box_b

#     inter_x1 = max(ax1, bx1)
#     inter_y1 = max(ay1, by1)
#     inter_x2 = min(ax2, bx2)
#     inter_y2 = min(ay2, by2)

#     inter_w = max(0, inter_x2 - inter_x1)
#     inter_h = max(0, inter_y2 - inter_y1)
#     inter_area = inter_w * inter_h

#     if inter_area == 0:
#         return 0.0

#     union = box_area(box_a) + box_area(box_b) - inter_area
#     return inter_area / union if union > 0 else 0.0


# def center_distance(box_a, box_b):
#     ax, ay = box_center(box_a)
#     bx, by = box_center(box_b)
#     return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# def contains_center(box_a, box_b):
#     x1, y1, x2, y2 = box_a
#     cx, cy = box_center(box_b)
#     return x1 <= cx <= x2 and y1 <= cy <= y2


# def dedupe_detections(detections, overlap_thresh=0.25, center_dist_thresh=90):
#     kept = {}

#     for tid, det in sorted(
#         detections.items(),
#         key=lambda item: (item[1]["conf"], box_area(item[1]["box"])),
#         reverse=True
#     ):
#         box = det["box"]
#         is_duplicate = False

#         for kept_det in kept.values():
#             kept_box = kept_det["box"]

#             same_region = (
#                 iou(box, kept_box) >= overlap_thresh
#                 or contains_center(box, kept_box)
#                 or contains_center(kept_box, box)
#                 or center_distance(box, kept_box) <= center_dist_thresh
#             )

#             if same_region:
#                 is_duplicate = True
#                 break

#         if not is_duplicate:
#             kept[tid] = det

#     return kept


# def xy(point):
#     return np.array([point[0], point[1]], dtype=np.float32)


# def dist_xy(a, b):
#     return float(np.linalg.norm(xy(a) - xy(b)))


# def unit_vec(v):
#     norm = float(np.linalg.norm(v))
#     if norm < 1e-9:
#         return None
#     return v / norm


# def angle_3pt(a, b, c):
#     a = xy(a)
#     b = xy(b)
#     c = xy(c)

#     v1 = a - b
#     v2 = c - b

#     denom = np.linalg.norm(v1) * np.linalg.norm(v2)

#     if denom < 1e-9:
#         return 0.0

#     cos_value = float(np.dot(v1, v2) / denom)
#     cos_value = max(-1.0, min(1.0, cos_value))

#     return math.degrees(math.acos(cos_value))


# def point_in_expanded_box(point, box, expand_ratio=0.25):
#     if box is None:
#         return False

#     x1, y1, x2, y2 = box
#     bw = x2 - x1
#     bh = y2 - y1

#     ex1 = x1 - bw * expand_ratio
#     ey1 = y1 - bh * expand_ratio
#     ex2 = x2 + bw * expand_ratio
#     ey2 = y2 + bh * expand_ratio

#     return ex1 <= point[0] <= ex2 and ey1 <= point[1] <= ey2


# def distance_to_box(point, box):
#     if box is None:
#         return None

#     px, py = point[0], point[1]
#     x1, y1, x2, y2 = box

#     dx = max(x1 - px, 0, px - x2)
#     dy = max(y1 - py, 0, py - y2)

#     return float((dx ** 2 + dy ** 2) ** 0.5)


# # ============================================================
# # ORIGINAL APPEARANCE / RE-LOCKING HELPERS
# # ============================================================

# def torso_crop(img, box):
#     x1, y1, x2, y2 = box
#     bw, bh = x2 - x1, y2 - y1

#     tx1 = x1 + int(bw * 0.2)
#     tx2 = x2 - int(bw * 0.2)
#     ty1 = y1 + int(bh * 0.25)
#     ty2 = y2 - int(bh * 0.35)

#     crop = img[max(0, ty1):ty2, max(0, tx1):tx2]
#     return crop


# def appearance_signature(img, box):
#     crop = torso_crop(img, box)

#     if crop.size == 0:
#         return None

#     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
#     cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

#     return hist


# def dark_torso_ratio(img, box):
#     crop = torso_crop(img, box)

#     if crop.size == 0:
#         return 1.0

#     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#     value = hsv[:, :, 2]
#     saturation = hsv[:, :, 1]

#     dark_mask = (value < 75) & (saturation < 110)
#     return float(np.mean(dark_mask))


# def hist_similarity(h1, h2):
#     if h1 is None or h2 is None:
#         return 0.0

#     return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


# def frame_signature(img):
#     small = cv2.resize(img, (160, 90))
#     gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

#     hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
#     cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

#     return hist


# def fighter_candidate_score(img, box, conf):
#     h, w = img.shape[:2]

#     area_norm = box_area(box) / float(w * h)
#     cx, cy = box_center(box)

#     center_x_norm = cx / w
#     center_y_norm = cy / h
#     bottom_norm = box[3] / h
#     top_norm = box[1] / h
#     height_norm = (box[3] - box[1]) / h

#     edge_penalty = abs(center_x_norm - 0.5) * 1.2
#     dark_penalty = dark_torso_ratio(img, box) * 0.45
#     high_in_frame_penalty = max(0.0, 0.42 - center_y_norm) * 3.5
#     top_penalty = max(0.0, 0.18 - top_norm) * 2.5
#     too_tall_penalty = max(0.0, height_norm - 0.62) * 2.5

#     return (
#         2.2 * area_norm
#         + 0.9 * bottom_norm
#         + 0.6 * center_y_norm
#         + 0.5 * conf
#         - edge_penalty
#         - dark_penalty
#         - high_in_frame_penalty
#         - top_penalty
#         - too_tall_penalty
#     )


# # ============================================================
# # PUNCH FEATURE HELPERS
# # ============================================================

# def get_arm_metrics(snapshot, arm):
#     keys = ARM_KEYS[arm]

#     shoulder = snapshot[keys["shoulder"]]
#     elbow = snapshot[keys["elbow"]]
#     wrist = snapshot[keys["wrist"]]
#     hip = snapshot[keys["hip"]]

#     visibility_min = min(
#         shoulder[3],
#         elbow[3],
#         wrist[3],
#         hip[3],
#     )

#     if visibility_min < MIN_ARM_VISIBILITY:
#         return None

#     torso_len = dist_xy(shoulder, hip)

#     if torso_len < 1e-6:
#         return None

#     extension = dist_xy(shoulder, wrist) / torso_len
#     elbow_angle = angle_3pt(shoulder, elbow, wrist)

#     return {
#         "shoulder": shoulder,
#         "elbow": elbow,
#         "wrist": wrist,
#         "hip": hip,
#         "visibility_min": float(visibility_min),
#         "torso_len": float(torso_len),
#         "extension": float(extension),
#         "elbow_angle": float(elbow_angle),
#         "wrist_relative_to_shoulder": xy(wrist) - xy(shoulder),
#         "opponent_center": snapshot.get("opponent_center"),
#         "opponent_box": snapshot.get("opponent_box"),
#     }


# def arm_metrics_json(metrics):
#     if metrics is None:
#         return None

#     return {
#         "visibility_min": metrics["visibility_min"],
#         "torso_len": metrics["torso_len"],
#         "extension": metrics["extension"],
#         "elbow_angle": metrics["elbow_angle"],
#         "shoulder": {
#             "x": float(metrics["shoulder"][0]),
#             "y": float(metrics["shoulder"][1]),
#             "z": float(metrics["shoulder"][2]),
#             "visibility": float(metrics["shoulder"][3]),
#         },
#         "elbow": {
#             "x": float(metrics["elbow"][0]),
#             "y": float(metrics["elbow"][1]),
#             "z": float(metrics["elbow"][2]),
#             "visibility": float(metrics["elbow"][3]),
#         },
#         "wrist": {
#             "x": float(metrics["wrist"][0]),
#             "y": float(metrics["wrist"][1]),
#             "z": float(metrics["wrist"][2]),
#             "visibility": float(metrics["wrist"][3]),
#         },
#         "hip": {
#             "x": float(metrics["hip"][0]),
#             "y": float(metrics["hip"][1]),
#             "z": float(metrics["hip"][2]),
#             "visibility": float(metrics["hip"][3]),
#         },
#     }


# def build_trajectory_json(snapshots, arm):
#     rows = []

#     prev_metrics = None

#     for snap in snapshots:
#         metrics = get_arm_metrics(snap, arm)

#         row = {
#             "frame": int(snap["frame"]),
#             "fighter_box": list(map(float, snap["fighter_box"])),
#             "opponent_box": list(map(float, snap["opponent_box"])) if snap.get("opponent_box") is not None else None,
#             "opponent_center": list(map(float, snap["opponent_center"])) if snap.get("opponent_center") is not None else None,
#             "arm_metrics": arm_metrics_json(metrics),
#             "frame_to_frame": None,
#         }

#         if metrics is not None and prev_metrics is not None:
#             rel_move = metrics["wrist_relative_to_shoulder"] - prev_metrics["wrist_relative_to_shoulder"]
#             rel_speed = float(np.linalg.norm(rel_move) / metrics["torso_len"])

#             extension_change = float(metrics["extension"] - prev_metrics["extension"])
#             elbow_angle_change = float(metrics["elbow_angle"] - prev_metrics["elbow_angle"])

#             row["frame_to_frame"] = {
#                 "relative_wrist_speed": rel_speed,
#                 "extension_change": extension_change,
#                 "elbow_angle_change": elbow_angle_change,
#             }

#         if metrics is not None:
#             prev_metrics = metrics

#         rows.append(row)

#     return rows


# # ============================================================
# # SIMPLE PUNCH DETECTOR
# # ============================================================

# class SimplePunchDetector:
#     """
#     Simple detector:
#     - Uses full-frame coordinates, not crop-normalized coordinates.
#     - Looks for a local extension peak.
#     - Requires wrist travel relative to shoulder.
#     - Uses opponent direction/position as a soft gate.
#     - Allows multiple punches close together.
#     """

#     def __init__(self):
#         self.history = deque(maxlen=LANDMARK_HISTORY_SIZE)
#         self.last_event_frame = {
#             "left": -10_000,
#             "right": -10_000,
#         }
#         self.prev_smoothed = None

#     def reset(self):
#         self.history.clear()
#         self.prev_smoothed = None

#     def smooth_snapshot(self, raw):
#         if self.prev_smoothed is None:
#             self.prev_smoothed = raw
#             return raw

#         smoothed = {
#             "frame": raw["frame"],
#             "fighter": raw["fighter"],
#             "fighter_box": raw["fighter_box"],
#             "opponent_box": raw["opponent_box"],
#             "opponent_center": raw["opponent_center"],
#         }

#         landmark_keys = [
#             "L_SHOULDER", "R_SHOULDER",
#             "L_ELBOW", "R_ELBOW",
#             "L_WRIST", "R_WRIST",
#             "L_HIP", "R_HIP",
#         ]

#         for key in landmark_keys:
#             cur = raw[key]
#             prev = self.prev_smoothed[key]

#             sx = (1 - SMOOTH_ALPHA) * prev[0] + SMOOTH_ALPHA * cur[0]
#             sy = (1 - SMOOTH_ALPHA) * prev[1] + SMOOTH_ALPHA * cur[1]
#             sz = (1 - SMOOTH_ALPHA) * prev[2] + SMOOTH_ALPHA * cur[2]

#             # Keep current visibility.
#             smoothed[key] = (float(sx), float(sy), float(sz), float(cur[3]))

#         self.prev_smoothed = smoothed
#         return smoothed

#     def push_landmarks(self, frame_count, fighter_id, landmarks, fighter_box, opponent_box=None):
#         x1, y1, x2, y2 = fighter_box
#         bw = x2 - x1
#         bh = y2 - y1

#         opponent_center = box_center(opponent_box) if opponent_box is not None else None

#         def full_frame_point(idx):
#             lm = landmarks[idx]

#             # Convert MediaPipe crop-normalized point into full-frame pixel coordinates.
#             px = lm.x * bw + x1
#             py = lm.y * bh + y1
#             pz = lm.z * bw

#             return (
#                 float(px),
#                 float(py),
#                 float(pz),
#                 float(lm.visibility),
#             )

#         raw_snapshot = {
#             "frame": int(frame_count),
#             "fighter": int(fighter_id),
#             "fighter_box": tuple(map(float, fighter_box)),
#             "opponent_box": tuple(map(float, opponent_box)) if opponent_box is not None else None,
#             "opponent_center": tuple(map(float, opponent_center)) if opponent_center is not None else None,

#             "L_SHOULDER": full_frame_point(L_SHOULDER),
#             "R_SHOULDER": full_frame_point(R_SHOULDER),

#             "L_ELBOW": full_frame_point(L_ELBOW),
#             "R_ELBOW": full_frame_point(R_ELBOW),

#             "L_WRIST": full_frame_point(L_WRIST),
#             "R_WRIST": full_frame_point(R_WRIST),

#             "L_HIP": full_frame_point(L_HIP),
#             "R_HIP": full_frame_point(R_HIP),
#         }

#         snapshot = self.smooth_snapshot(raw_snapshot)
#         self.history.append(snapshot)

#     def detect(self):
#         events = []
#         hist = list(self.history)

#         needed = PUNCH_LOOKBACK_FRAMES + PUNCH_CONFIRM_AFTER + 1

#         if len(hist) < needed:
#             return events

#         peak_idx = len(hist) - 1 - PUNCH_CONFIRM_AFTER

#         if peak_idx < PUNCH_LOOKBACK_FRAMES:
#             return events

#         peak_snap = hist[peak_idx]
#         peak_frame = peak_snap["frame"]

#         pre_snaps = hist[peak_idx - PUNCH_LOOKBACK_FRAMES:peak_idx]
#         post_snaps = hist[peak_idx + 1:peak_idx + PUNCH_CONFIRM_AFTER + 1]
#         local_snaps = pre_snaps + [peak_snap] + post_snaps

#         for arm in ["left", "right"]:
#             if peak_frame - self.last_event_frame[arm] < SAME_ARM_DEBOUNCE_FRAMES:
#                 continue

#             peak_metrics = get_arm_metrics(peak_snap, arm)

#             if peak_metrics is None:
#                 continue

#             pre_metrics = []
#             for snap in pre_snaps:
#                 m = get_arm_metrics(snap, arm)
#                 if m is not None:
#                     pre_metrics.append((snap, m))

#             post_metrics = []
#             for snap in post_snaps:
#                 m = get_arm_metrics(snap, arm)
#                 if m is not None:
#                     post_metrics.append((snap, m))

#             local_metrics = []
#             for snap in local_snaps:
#                 m = get_arm_metrics(snap, arm)
#                 if m is not None:
#                     local_metrics.append((snap, m))

#             if len(pre_metrics) < 3 or len(post_metrics) < 1 or len(local_metrics) < 5:
#                 continue

#             # Start of punch motion = lowest extension before peak.
#             start_snap, start_metrics = min(pre_metrics, key=lambda item: item[1]["extension"])
#             start_frame = start_snap["frame"]

#             duration_frames = peak_frame - start_frame

#             if duration_frames <= 0:
#                 continue

#             local_extensions = [m["extension"] for _, m in local_metrics]
#             max_local_extension = max(local_extensions)

#             # Peak must be close to the local max.
#             is_local_peak = peak_metrics["extension"] >= max_local_extension - 0.025

#             if not is_local_peak:
#                 continue

#             extension_delta = peak_metrics["extension"] - start_metrics["extension"]

#             relative_wrist_move = (
#                 peak_metrics["wrist_relative_to_shoulder"]
#                 - start_metrics["wrist_relative_to_shoulder"]
#             )

#             rel_wrist_travel = float(np.linalg.norm(relative_wrist_move) / peak_metrics["torso_len"])
#             abs_wrist_travel = dist_xy(peak_metrics["wrist"], start_metrics["wrist"]) / peak_metrics["torso_len"]
#             rel_wrist_speed = rel_wrist_travel / max(1, duration_frames)

#             elbow_angle_delta = peak_metrics["elbow_angle"] - start_metrics["elbow_angle"]

#             post_extension_min = min([m["extension"] for _, m in post_metrics])
#             post_extension_drop = peak_metrics["extension"] - post_extension_min

#             opponent_center = peak_metrics["opponent_center"]
#             opponent_box = peak_metrics["opponent_box"]

#             toward_score = None
#             opponent_distance_delta = None
#             wrist_distance_to_opponent_box = None
#             wrist_near_opponent_box = False

#             if opponent_center is not None:
#                 move_unit = unit_vec(relative_wrist_move)

#                 opponent_vector = xy(opponent_center) - xy(start_metrics["shoulder"])
#                 opponent_unit = unit_vec(opponent_vector)

#                 if move_unit is not None and opponent_unit is not None:
#                     toward_score = float(np.dot(move_unit, opponent_unit))

#                 start_dist_to_opp = float(np.linalg.norm(xy(start_metrics["wrist"]) - xy(opponent_center)))
#                 peak_dist_to_opp = float(np.linalg.norm(xy(peak_metrics["wrist"]) - xy(opponent_center)))

#                 opponent_distance_delta = (start_dist_to_opp - peak_dist_to_opp) / peak_metrics["torso_len"]

#                 wrist_distance_to_opponent_box = distance_to_box(peak_metrics["wrist"], opponent_box)
#                 wrist_near_opponent_box = point_in_expanded_box(
#                     peak_metrics["wrist"],
#                     opponent_box,
#                     expand_ratio=0.30
#                 )

#             # Opponent gate:
#             # If opponent is available, reject movements clearly away from opponent.
#             # But don't make this too strict because hooks and camera angles are messy.
#             opponent_gate = True

#             if opponent_center is not None:
#                 not_clearly_away = (
#                     toward_score is None
#                     or toward_score >= -0.25
#                 )

#                 not_getting_much_farther = (
#                     opponent_distance_delta is None
#                     or opponent_distance_delta >= -0.20
#                 )

#                 opponent_gate = (
#                     wrist_near_opponent_box
#                     or (not_clearly_away and not_getting_much_farther)
#                 )

#             is_punch = (
#                 opponent_gate
#                 and 2 <= duration_frames <= 12
#                 and extension_delta >= MIN_EXTENSION_DELTA
#                 and peak_metrics["extension"] >= MIN_PEAK_EXTENSION
#                 and rel_wrist_travel >= MIN_REL_WRIST_TRAVEL
#                 and abs_wrist_travel >= MIN_ABS_WRIST_TRAVEL
#                 and rel_wrist_speed >= MIN_REL_WRIST_SPEED
#                 and peak_metrics["elbow_angle"] >= MIN_ELBOW_ANGLE_AT_PEAK
#             )

#             if not is_punch:
#                 continue

#             features = {
#                 "peak_frame": int(peak_frame),
#                 "start_frame": int(start_frame),
#                 "duration_frames": int(duration_frames),

#                 "extension_start": float(start_metrics["extension"]),
#                 "extension_peak": float(peak_metrics["extension"]),
#                 "extension_delta": float(extension_delta),
#                 "post_extension_drop": float(post_extension_drop),

#                 "relative_wrist_travel": float(rel_wrist_travel),
#                 "absolute_wrist_travel": float(abs_wrist_travel),
#                 "relative_wrist_speed": float(rel_wrist_speed),

#                 "elbow_angle_start": float(start_metrics["elbow_angle"]),
#                 "elbow_angle_peak": float(peak_metrics["elbow_angle"]),
#                 "elbow_angle_delta": float(elbow_angle_delta),

#                 "visibility_min_peak": float(peak_metrics["visibility_min"]),
#                 "torso_len_peak": float(peak_metrics["torso_len"]),

#                 "toward_score": None if toward_score is None else float(toward_score),
#                 "opponent_distance_delta": None if opponent_distance_delta is None else float(opponent_distance_delta),
#                 "wrist_distance_to_opponent_box": None if wrist_distance_to_opponent_box is None else float(wrist_distance_to_opponent_box),
#                 "wrist_near_opponent_box": bool(wrist_near_opponent_box),
#                 "opponent_gate": bool(opponent_gate),
#             }

#             events.append({
#                 "fighter": int(peak_snap["fighter"]),
#                 "arm": arm,
#                 "strike_type": "punch_candidate",
#                 "peak_frame": int(peak_frame),
#                 "features": features,
#             })

#             self.last_event_frame[arm] = peak_frame

#         return events


# # ============================================================
# # FIGHTER TRACKER
# # ============================================================

# class FighterTracker:
#     def __init__(self, fighter_id):
#         self.fighter_id = fighter_id
#         self.pose = mpPose.Pose(
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.data = []
#         self.signature = None
#         self.punch_detector = SimplePunchDetector()

#     def reset_punch_detector(self):
#         self.punch_detector.reset()

#     def update_signature(self, new_sig):
#         if new_sig is None:
#             return

#         if self.signature is None:
#             self.signature = new_sig
#         else:
#             self.signature = (1 - SIG_UPDATE_WEIGHT) * self.signature + SIG_UPDATE_WEIGHT * new_sig

#     def process(self, img, box, frame_count, opponent_box=None, detect_punches=True):
#         x1, y1, x2, y2 = box
#         crop = img[y1:y2, x1:x2]

#         if crop.size == 0:
#             return []

#         ch, cw = crop.shape[:2]

#         results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

#         if not results.pose_landmarks:
#             return []

#         lm_list = results.pose_landmarks.landmark

#         # Store full-frame smoothed landmarks for punch detection.
#         self.punch_detector.push_landmarks(
#             frame_count=frame_count,
#             fighter_id=self.fighter_id,
#             landmarks=lm_list,
#             fighter_box=box,
#             opponent_box=opponent_box,
#         )

#         punch_events = self.punch_detector.detect() if detect_punches else []

#         frame_data = {
#             "frame": frame_count,
#             "fighter": self.fighter_id,
#             "landmarks": [],
#         }

#         for landmark_id, lm in enumerate(lm_list):
#             if landmark_id not in excluded_landmarks:
#                 cx = int(lm.x * cw) + x1
#                 cy = int(lm.y * ch) + y1

#                 color = (255, 0, 0) if self.fighter_id == 0 else (0, 0, 255)
#                 cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

#                 if lm.visibility > 0.35:
#                     frame_data["landmarks"].append({
#                         "id": landmark_id,
#                         "x": lm.x,
#                         "y": lm.y,
#                         "z": lm.z,
#                         "visibility": lm.visibility,
#                     })

#         for connection in mpPose.POSE_CONNECTIONS:
#             if connection[0] not in excluded_landmarks and connection[1] not in excluded_landmarks:
#                 start = lm_list[connection[0]]
#                 end = lm_list[connection[1]]

#                 sx = int(start.x * cw) + x1
#                 sy = int(start.y * ch) + y1
#                 ex = int(end.x * cw) + x1
#                 ey = int(end.y * ch) + y1

#                 color = (0, 255, 0) if self.fighter_id == 0 else (0, 165, 255)
#                 cv2.line(img, (sx, sy), (ex, ey), color, 2)

#         for ev in punch_events:
#             arm = ev["arm"]
#             wrist_idx = L_WRIST if arm == "left" else R_WRIST

#             wlm = lm_list[wrist_idx]
#             wx = int(wlm.x * cw) + x1
#             wy = int(wlm.y * ch) + y1

#             cv2.circle(img, (wx, wy), 18, (0, 255, 255), 3)
#             cv2.putText(
#                 img,
#                 f"PUNCH {arm[0].upper()}",
#                 (wx + 18, wy),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65,
#                 (0, 255, 255),
#                 2,
#             )

#         self.data.append(frame_data)

#         return punch_events


# # ============================================================
# # LOCKING / RE-LOCKING
# # ============================================================

# def lock_to_biggest(detections, img):
#     if len(detections) < 2:
#         return None

#     candidates = sorted(
#         detections.items(),
#         key=lambda d: d[1]["fighter_score"],
#         reverse=True,
#     )

#     best_pair = None
#     best_score = -1.0

#     for i, (tid_a, det_a) in enumerate(candidates):
#         box_a = det_a["box"]

#         for tid_b, det_b in candidates[i + 1:]:
#             box_b = det_b["box"]

#             overlap = iou(box_a, box_b)
#             separation = abs(box_center_x(box_a) - box_center_x(box_b))

#             if overlap >= 0.20 or separation < img.shape[1] * MIN_SEPARATION_RATIO:
#                 continue

#             xa = box_center_x(box_a) / img.shape[1]
#             xb = box_center_x(box_b) / img.shape[1]

#             if (xa < 0.5 and xb < 0.5) or (xa > 0.5 and xb > 0.5):
#                 continue

#             pair_score = (
#                 1200.0 * (det_a["fighter_score"] + det_b["fighter_score"])
#                 + box_area(box_a)
#                 + box_area(box_b)
#                 + separation
#                 + 1000.0 * (det_a["conf"] + det_b["conf"])
#             )

#             if pair_score > best_score:
#                 left, right = ((tid_a, box_a), (tid_b, box_b))

#                 if box_center_x(box_a) > box_center_x(box_b):
#                     left, right = right, left

#                 best_pair = [left, right]
#                 best_score = pair_score

#     if best_pair is None:
#         return None

#     result = []

#     for tid, box in best_pair:
#         sig = appearance_signature(img, box)
#         result.append((tid, box, sig))

#     return result


# def relock_by_appearance(trackers, detections, img):
#     if not detections:
#         return {}

#     candidates = []

#     for fi, tracker in enumerate(trackers):
#         if tracker.signature is None:
#             continue

#         for tid, det in detections.items():
#             box = det["box"]
#             sig = appearance_signature(img, box)

#             score = hist_similarity(tracker.signature, sig) + 0.35 * det["fighter_score"]
#             candidates.append((score, fi, tid))

#     candidates.sort(reverse=True, key=lambda x: x[0])

#     assigned_fighters = set()
#     assigned_tids = set()
#     new_locks = {}

#     for score, fi, tid in candidates:
#         if score < MIN_MATCH_SCORE:
#             break

#         if fi in assigned_fighters or tid in assigned_tids:
#             continue

#         new_locks[fi] = tid
#         assigned_fighters.add(fi)
#         assigned_tids.add(tid)

#     return new_locks


# # ============================================================
# # SAVING CLIPS + JSON
# # ============================================================

# def next_clip_index(base_dir):
#     os.makedirs(base_dir, exist_ok=True)

#     max_idx = 0

#     for name in os.listdir(base_dir):
#         if not name.startswith("video_clip_"):
#             continue

#         suffix = name.replace("video_clip_", "")

#         try:
#             idx = int(suffix)
#             max_idx = max(max_idx, idx)
#         except ValueError:
#             pass

#     return max_idx + 1


# def save_punch_event(
#     event,
#     tracker,
#     raw_frame_buffer,
#     clip_counter,
#     video_fps,
#     frame_w,
#     frame_h,
# ):
#     peak_frame = event["peak_frame"]
#     arm = event["arm"]

#     win_start = peak_frame - CLIP_FRAMES_BEFORE
#     win_end = peak_frame + CLIP_FRAMES_AFTER

#     frame_items = [
#         (fc, frame)
#         for fc, frame in raw_frame_buffer
#         if win_start <= fc <= win_end
#     ]

#     if len(frame_items) < 5:
#         print(f"  -> SKIP save: only {len(frame_items)} raw frames for frame {peak_frame}")
#         return clip_counter

#     history_items = [
#         snap
#         for snap in tracker.punch_detector.history
#         if win_start <= snap["frame"] <= win_end
#     ]

#     folder_name = f"video_clip_{clip_counter:06d}"
#     folder_path = os.path.join(DATA_COLLECTION_DIR, folder_name)
#     os.makedirs(folder_path, exist_ok=True)

#     frame_items.sort(key=lambda x: x[0])

#     clip_path = os.path.join(folder_path, "clip.mp4")
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(clip_path, fourcc, video_fps, (frame_w, frame_h))

#     if not out.isOpened():
#         print(f"  -> ERROR: could not write {clip_path}")
#         return clip_counter

#     for _, frame in frame_items:
#         out.write(frame)

#     out.release()

#     json_record = {
#         "video_path": VIDEO_PATH,
#         "clip_folder": folder_name,
#         "clip_file": "clip.mp4",

#         "fighter": event["fighter"],
#         "arm": arm,
#         "strike_type": event["strike_type"],

#         "peak_frame": peak_frame,
#         "window_start": win_start,
#         "window_end": win_end,
#         "raw_clip_frame_count": len(frame_items),

#         "label": "unlabeled",
#         "punch": True,

#         "decision_features": event["features"],

#         # This is useful later for Random Forest or debugging.
#         # It includes the per-frame measurements around the punch.
#         "trajectory": build_trajectory_json(history_items, arm),
#     }

#     json_path = os.path.join(folder_path, "features.json")

#     with open(json_path, "w") as f:
#         json.dump(json_record, f, indent=2)

#     print(
#         f"  -> SAVED {folder_name}: fighter={event['fighter']} arm={arm} "
#         f"peak={peak_frame} ext_delta={event['features']['extension_delta']:.3f} "
#         f"travel={event['features']['relative_wrist_travel']:.3f} "
#         f"speed={event['features']['relative_wrist_speed']:.3f}"
#     )

#     return clip_counter + 1


# # ============================================================
# # BYTETRACK CONFIG
# # ============================================================

# BYTETRACK_CONFIG = """
# tracker_type: bytetrack
# track_high_thresh: 0.5
# track_low_thresh: 0.1
# new_track_thresh: 0.5
# track_buffer: 60
# match_thresh: 0.7
# fuse_score: True
# """

# with open("custom_bytetrack.yaml", "w") as f:
#     f.write(BYTETRACK_CONFIG)


# # ============================================================
# # MAIN
# # ============================================================

# os.makedirs(DATA_COLLECTION_DIR, exist_ok=True)

# clip_counter = next_clip_index(DATA_COLLECTION_DIR)

# cap = cv2.VideoCapture(VIDEO_PATH)

# if not cap.isOpened():
#     raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

# model = YOLO(MODEL_PATH)

# video_fps = cap.get(cv2.CAP_PROP_FPS)

# if video_fps <= 0:
#     video_fps = 30

# frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# trackers = [
#     FighterTracker(0),
#     FighterTracker(1),
# ]

# locked_ids = [None, None]

# prev_frame_sig = None
# missing_streak = 0
# fighter_missing_streaks = [0, 0]
# frame_count = 0

# ignore_punch_until_frame = -1

# raw_frame_buffer = deque(maxlen=RAW_FRAME_BUFFER_SIZE)

# pending_events = []

# while True:
#     success, img = cap.read()

#     if not success:
#         break

#     raw_frame_buffer.append((frame_count, img.copy()))

#     h, w = img.shape[:2]

#     # ------------------------------------------------------------
#     # Scene cut detection
#     # ------------------------------------------------------------
#     cur_frame_sig = frame_signature(img)
#     scene_cut = False

#     if prev_frame_sig is not None:
#         sim = cv2.compareHist(prev_frame_sig, cur_frame_sig, cv2.HISTCMP_CORREL)

#         if sim < SCENE_CUT_THRESHOLD:
#             scene_cut = True

#     prev_frame_sig = cur_frame_sig

#     if scene_cut:
#         ignore_punch_until_frame = frame_count + SCENE_CUT_IGNORE_FRAMES

#         for tracker in trackers:
#             tracker.reset_punch_detector()

#         print(
#             f"[frame {frame_count}] SCENE CUT: ignoring punch detection until "
#             f"frame {ignore_punch_until_frame}"
#         )

#     # ------------------------------------------------------------
#     # YOLO + ByteTrack
#     # ------------------------------------------------------------
#     results = model.track(
#         img,
#         classes=[0],
#         conf=0.35,
#         persist=True,
#         verbose=False,
#         tracker="custom_bytetrack.yaml"
#     )

#     detections = {}

#     if results[0].boxes is not None and results[0].boxes.id is not None:
#         for box, track_id, conf in zip(
#             results[0].boxes.xyxy,
#             results[0].boxes.id,
#             results[0].boxes.conf
#         ):
#             x1, y1, x2, y2 = map(int, box)

#             x1 = max(0, min(x1, w - 1))
#             y1 = max(0, min(y1, h - 1))
#             x2 = max(0, min(x2, w - 1))
#             y2 = max(0, min(y2, h - 1))

#             if x2 <= x1 or y2 <= y1:
#                 continue

#             tid = int(track_id)
#             cur_box = (x1, y1, x2, y2)

#             if box_area(cur_box) < 0.04 * w * h:
#                 continue

#             score = fighter_candidate_score(img, cur_box, float(conf))

#             if score < 0.12:
#                 continue

#             detections[tid] = {
#                 "box": cur_box,
#                 "conf": float(conf),
#                 "fighter_score": score,
#             }

#     detections = dedupe_detections(detections)

#     # ------------------------------------------------------------
#     # Initial lock
#     # ------------------------------------------------------------
#     if all(lid is None for lid in locked_ids) and frame_count < INITIAL_LOCK_FRAMES:
#         locks = lock_to_biggest(detections, img)

#         if locks is not None:
#             for i, (tid, box, sig) in enumerate(locks):
#                 locked_ids[i] = tid
#                 trackers[i].signature = sig

#             print(f"[frame {frame_count}] Initial lock: {locked_ids}")

#     # ------------------------------------------------------------
#     # Missing / unlock logic
#     # ------------------------------------------------------------
#     locked_present = [lid is not None for lid in locked_ids]
#     visible = [lid is not None and lid in detections for lid in locked_ids]

#     for i, lid in enumerate(locked_ids):
#         if lid is None:
#             fighter_missing_streaks[i] = 0
#             continue

#         if lid in detections:
#             fighter_missing_streaks[i] = 0
#         else:
#             fighter_missing_streaks[i] += 1

#             if fighter_missing_streaks[i] >= SINGLE_FIGHTER_MISSING_UNLOCK:
#                 print(f"[frame {frame_count}] Unlock fighter {i}: lost track {lid}")
#                 locked_ids[i] = None
#                 fighter_missing_streaks[i] = 0

#     if any(locked_present) and not any(visible):
#         missing_streak += 1
#     else:
#         missing_streak = 0

#     need_relock = (
#         any(lid is None for lid in locked_ids)
#         or scene_cut
#         or missing_streak >= MISSING_FRAMES_RELOCK
#     )

#     if need_relock and frame_count >= INITIAL_LOCK_FRAMES and len(detections) >= 1:
#         new_locks = relock_by_appearance(trackers, detections, img)

#         if new_locks:
#             for fi, tid in new_locks.items():
#                 if tid in locked_ids and locked_ids[fi] != tid:
#                     continue

#                 if locked_ids[fi] != tid:
#                     reason = "scene cut" if scene_cut else ("missing" if missing_streak else "unlocked")
#                     print(f"[frame {frame_count}] Re-lock fighter {fi}: {locked_ids[fi]} -> {tid} ({reason})")

#                 locked_ids[fi] = tid

#             missing_streak = 0

#     # ------------------------------------------------------------
#     # Process locked fighters + punch detection
#     # ------------------------------------------------------------
#     detect_punches_this_frame = frame_count > ignore_punch_until_frame

#     for i, fid in enumerate(locked_ids):
#         if fid is None or fid not in detections:
#             continue

#         box = detections[fid]["box"]
#         x1, y1, x2, y2 = box

#         color = (0, 255, 0) if i == 0 else (0, 0, 255)

#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(
#             img,
#             f"Fighter {i + 1}",
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             color,
#             2,
#         )

#         opponent_box = None
#         other_i = 1 - i
#         other_fid = locked_ids[other_i]

#         if other_fid is not None and other_fid in detections:
#             opponent_box = detections[other_fid]["box"]

#         punch_events = trackers[i].process(
#             img=img,
#             box=box,
#             frame_count=frame_count,
#             opponent_box=opponent_box,
#             detect_punches=detect_punches_this_frame,
#         )

#         trackers[i].update_signature(appearance_signature(img, box))

#         for ev in punch_events:
#             save_at = ev["peak_frame"] + CLIP_FRAMES_AFTER

#             pending_events.append({
#                 "event": ev,
#                 "tracker_index": i,
#                 "save_at": save_at,
#             })

#             print(
#                 f"[frame {frame_count}] PUNCH CANDIDATE: fighter={i} "
#                 f"arm={ev['arm']} peak={ev['peak_frame']} save_at={save_at} "
#                 f"features={ev['features']}"
#             )

#     # ------------------------------------------------------------
#     # Save pending clips once future frames are available
#     # ------------------------------------------------------------
#     still_pending = []

#     for pending in pending_events:
#         if frame_count < pending["save_at"]:
#             still_pending.append(pending)
#             continue

#         tracker_idx = pending["tracker_index"]
#         event = pending["event"]

#         clip_counter = save_punch_event(
#             event=event,
#             tracker=trackers[tracker_idx],
#             raw_frame_buffer=raw_frame_buffer,
#             clip_counter=clip_counter,
#             video_fps=video_fps,
#             frame_w=frame_w,
#             frame_h=frame_h,
#         )

#     pending_events = still_pending

#     # ------------------------------------------------------------
#     # HUD
#     # ------------------------------------------------------------
#     if scene_cut:
#         cv2.putText(
#             img,
#             "SCENE CUT",
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.0,
#             (0, 255, 255),
#             2,
#         )

#     if not detect_punches_this_frame:
#         cv2.putText(
#             img,
#             "IGNORING PUNCHES AFTER CUT",
#             (20, 75),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 255),
#             2,
#         )

#     status = (
#         f"F1={locked_ids[0]} F2={locked_ids[1]} "
#         f"pending={len(pending_events)} next_clip={clip_counter:06d}"
#     )

#     cv2.putText(
#         img,
#         status,
#         (20, h - 20),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (255, 255, 255),
#         2,
#     )

#     cv2.imshow("Fighters", img)

#     if cv2.waitKey(max(1, int(1000 / video_fps))) & 0xFF == ord("q"):
#         break

#     frame_count += 1


# # ============================================================
# # END-OF-VIDEO FLUSH
# # ============================================================

# for pending in pending_events:
#     tracker_idx = pending["tracker_index"]
#     event = pending["event"]

#     clip_counter = save_punch_event(
#         event=event,
#         tracker=trackers[tracker_idx],
#         raw_frame_buffer=raw_frame_buffer,
#         clip_counter=clip_counter,
#         video_fps=video_fps,
#         frame_w=frame_w,
#         frame_h=frame_h,
#     )


# # ============================================================
# # SAVE FULL LANDMARK LOGS
# # ============================================================

# for tracker in trackers:
#     with open(f"landmarks_fighter_{tracker.fighter_id}.json", "w") as f:
#         json.dump(tracker.data, f, indent=4)

#     print(f"Fighter {tracker.fighter_id}: {len(tracker.data)} frames captured")


# cap.release()
# cv2.destroyAllWindows()
