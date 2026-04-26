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
SINGLE_FIGHTER_MISSING_UNLOCK = 6  # if one fighter is hidden this long, unlock that slot so it can reacquire
MIN_MATCH_SCORE = 0.5          # min histogram correlation to accept a re-lock match
SIG_UPDATE_WEIGHT = 0.1        # EMA weight for slowly refreshing stored signatures
INITIAL_LOCK_FRAMES = 20       # frames to wait before locking initially
MIN_SEPARATION_RATIO = 0.18    # fighter centers should be meaningfully separated left/right

excluded_landmarks = list(range(0, 11)) + list(range(25, 33))
mpPose = mp.solutions.pose


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def box_center_x(box):
    x1, _, x2, _ = box
    return (x1 + x2) / 2.0


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union = box_area(box_a) + box_area(box_b) - inter_area
    return inter_area / union if union > 0 else 0.0


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def contains_center(box_a, box_b):
    """
    True if the center of box_b lies inside box_a.
    Useful when one duplicate box is much larger, making IoU misleadingly low.
    """
    x1, y1, x2, y2 = box_a
    cx, cy = box_center(box_b)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def dedupe_detections(detections, overlap_thresh=0.25, center_dist_thresh=90):
    """
    Remove near-duplicate boxes so one fighter does not occupy multiple track IDs.
    Keeps the highest-confidence boxes first.
    """
    kept = {}
    for tid, det in sorted(detections.items(), key=lambda item: (item[1]["conf"], box_area(item[1]["box"])), reverse=True):
        box = det["box"]
        is_duplicate = False
        for kept_det in kept.values():
            kept_box = kept_det["box"]
            same_region = (
                iou(box, kept_box) >= overlap_thresh or
                contains_center(box, kept_box) or
                contains_center(kept_box, box) or
                center_distance(box, kept_box) <= center_dist_thresh
            )
            if same_region:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        kept[tid] = det
    return kept


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


def dark_torso_ratio(img, box):
    """
    Rough referee heuristic: refs often wear dark shirts, while fighters show more skin/shorts.
    Returns fraction of torso pixels that are very dark.
    """
    crop = torso_crop(img, box)
    if crop.size == 0:
        return 1.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    dark_mask = (value < 75) & (saturation < 110)
    return float(np.mean(dark_mask))


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


def fighter_candidate_score(img, box, conf):
    """
    Favor active fighters over the referee:
    - larger area
    - lower in the frame (fighters occupy more of the canvas)
    - closer to center than the cage edge
    - less likely to be wearing a dark referee shirt
    """
    h, w = img.shape[:2]
    area_norm = box_area(box) / float(w * h)
    cx, cy = box_center(box)
    center_x_norm = cx / w
    center_y_norm = cy / h
    bottom_norm = box[3] / h
    top_norm = box[1] / h
    height_norm = (box[3] - box[1]) / h
    edge_penalty = abs(center_x_norm - 0.5) * 1.2
    dark_penalty = dark_torso_ratio(img, box) * 0.45
    high_in_frame_penalty = max(0.0, 0.42 - center_y_norm) * 3.5
    top_penalty = max(0.0, 0.18 - top_norm) * 2.5
    too_tall_penalty = max(0.0, height_norm - 0.62) * 2.5

    return (
        2.2 * area_norm +
        0.9 * bottom_norm +
        0.6 * center_y_norm +
        0.5 * conf -
        edge_penalty -
        dark_penalty -
        high_in_frame_penalty -
        top_penalty -
        too_tall_penalty
    )

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
    """Initial lock: pick distinct left/right fighters, avoiding overlapping duplicates."""
    if len(detections) < 2:
        return None

    candidates = sorted(
        detections.items(),
        key=lambda d: d[1]["fighter_score"],
        reverse=True,
    )
    best_pair = None
    best_score = -1.0

    for i, (tid_a, det_a) in enumerate(candidates):
        box_a = det_a["box"]
        for tid_b, det_b in candidates[i + 1:]:
            box_b = det_b["box"]
            overlap = iou(box_a, box_b)
            separation = abs(box_center_x(box_a) - box_center_x(box_b))
            if overlap >= 0.20 or separation < img.shape[1] * MIN_SEPARATION_RATIO:
                continue

            # Avoid selecting two people from the same side of the frame.
            xa = box_center_x(box_a) / img.shape[1]
            xb = box_center_x(box_b) / img.shape[1]
            if (xa < 0.5 and xb < 0.5) or (xa > 0.5 and xb > 0.5):
                continue

            # Favor strong, distinct left/right candidates over a single dominant fighter.
            pair_score = (
                1200.0 * (det_a["fighter_score"] + det_b["fighter_score"]) +
                box_area(box_a) + box_area(box_b) +
                separation +
                1000.0 * (det_a["conf"] + det_b["conf"])
            )
            if pair_score > best_score:
                left, right = ((tid_a, box_a), (tid_b, box_b))
                if box_center_x(box_a) > box_center_x(box_b):
                    left, right = right, left
                best_pair = [left, right]
                best_score = pair_score

    if best_pair is None:
        return None

    result = []
    for tid, box in best_pair:
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
        for tid, det in detections.items():
            box = det["box"]
            sig = appearance_signature(img, box)
            score = hist_similarity(tracker.signature, sig) + 0.35 * det["fighter_score"]
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
fighter_missing_streaks = [0, 0]
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
        for box, track_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            tid = int(track_id)
            cur_box = (x1, y1, x2, y2)
            if box_area(cur_box) < 0.04 * w * h:
                continue
            score = fighter_candidate_score(img, cur_box, float(conf))
            if score < 0.12:
                continue
            detections[tid] = {"box": cur_box, "conf": float(conf), "fighter_score": score}

    detections = dedupe_detections(detections)

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

    for i, lid in enumerate(locked_ids):
        if lid is None:
            fighter_missing_streaks[i] = 0
            continue
        if lid in detections:
            fighter_missing_streaks[i] = 0
        else:
            fighter_missing_streaks[i] += 1
            if fighter_missing_streaks[i] >= SINGLE_FIGHTER_MISSING_UNLOCK:
                print(f"[frame {frame_count}] Unlock fighter {i}: lost track {lid}")
                locked_ids[i] = None
                fighter_missing_streaks[i] = 0

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
                # Do not allow both fighters to collapse onto the same track ID.
                if tid in locked_ids and locked_ids[fi] != tid:
                    continue
                if locked_ids[fi] != tid:
                    reason = 'scene cut' if scene_cut else ('missing' if missing_streak else 'unlocked')
                    print(f"[frame {frame_count}] Re-lock fighter {fi}: {locked_ids[fi]} -> {tid} ({reason})")
                locked_ids[fi] = tid
            missing_streak = 0

    # --- Process locked fighters ---
    for i, fid in enumerate(locked_ids):
        if fid is None or fid not in detections:
            continue
        box = detections[fid]["box"]
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
