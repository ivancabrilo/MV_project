import cv2
import mediapipe as mp
import json
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "test_video4.mp4"
MODEL_PATH = "yolo26n.pt"

# --- Re-locking tuning knobs ---
SCENE_CUT_THRESHOLD = 0.6      # histogram correlation below this = camera cut
MISSING_FRAMES_RELOCK = 10     # both fighters missing this long = trigger re-lock (fallback)
MIN_MATCH_SCORE = 0.5          # min histogram correlation to accept a re-lock match
SIG_UPDATE_WEIGHT = 0.1        # EMA weight for slowly refreshing stored signatures
INITIAL_LOCK_FRAMES = 20       # frames to wait before locking initially

excluded_landmarks = list(range(0, 11)) + list(range(25, 33))
mpPose = mp.solutions.pose


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def torso_crop(img, box):
    """Take the middle 60% of the bounding box (mostly torso, avoids head/legs/background)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1 = x1 + int(bw * 0.2)
    tx2 = x2 - int(bw * 0.2)
    ty1 = y1 + int(bh * 0.25)
    ty2 = y2 - int(bh * 0.35)
    crop = img[max(0, ty1):ty2, max(0, tx1):tx2]
    return crop


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


def frame_signature(img):
    """Coarse grayscale histogram of the whole frame, for scene-cut detection."""
    small = cv2.resize(img, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


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

        results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return

        frame_data = {"frame": frame_count, "fighter": self.fighter_id, "landmarks": []}

        for id, lm in enumerate(results.pose_landmarks.landmark):
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