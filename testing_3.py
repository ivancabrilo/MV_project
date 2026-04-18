import cv2
import mediapipe as mp
import json
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "test_video.mp4"
MODEL_PATH = "yolo26s.pt"

# All these are re-lock related
SCENE_CUT_THRESHOLD = 0.6      # histogram correlation below this = camera cut (camera view changed suddenly)
MISSING_FRAMES_RELOCK = 7     # both fighters missing this long = trigger re-lock (fallback)
MIN_MATCH_SCORE = 0.5          # min histogram correlation to accept a re-lock match
SIG_UPDATE_WEIGHT = 0.1        # Exponential Moving Average (EMA) weight: how fast the live signature adapts
SIG_UPDATE_MIN_CORR = 0.7      # new crop must match stored signiture this well to allow update
INITIAL_LOCK_FRAMES = 20       # frames to wait before locking initially
MIN_BOX_CENTER_DIST = 0.12     # NEW: min horizontal distance between centers of locked boxes (fraction of frame width)

excluded_landmarks = list(range(0, 11)) + list(range(25, 33)) # we don't care abt these points
mpPose = mp.solutions.pose # google mediapipe object

# very important
# used to disregard ppl in audiance 
# also used in lock_to_biggest to pick two largest boxes as fighters
def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

# returns central cooridantes of a box
def box_center(box):
    """Returns (cx, cy) of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# used to reject inital locks if fighters too close (detection overlap > 0.3)
def box_iou(box_a, box_b):
    """Intersection over Union of two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

# we maintain the fighter tracker based on their shorts color
def shorts_crop(img, box):
    """Crop focused on the shorts region only, since it's the most distinctive part of each fighter."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1 = x1 + int(bw * 0.15)
    tx2 = x2 - int(bw * 0.15)
    ty1 = y1 + int(bh * 0.45)   # start below the waist
    ty2 = y2 - int(bh * 0.25)   # stop above the knees
    crop = img[max(0, ty1):ty2, max(0, tx1):tx2]
    return crop


def appearance_signature(img, box):
    """HSV color histogram of the shorts region. Simple and focused."""
    crop = shorts_crop(img, box)
    if crop.size == 0:
        return None
    if crop.shape[0] < 5 or crop.shape[1] < 5:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

# compare how similar 2 signitures are
# does this new detection's shorts look like fighter 1's stored shorts
def hist_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

# if correlation between 2 frames is too low, we know there has been a switch of scene
def frame_signature(img):
    """Coarse grayscale histogram of the whole frame, for scene-cut detection."""
    small = cv2.resize(img, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


# Guarded signature updates + frozen reference copy
# - reference_sig: snapshot from initial lock, NEVER updated. Anchor for re-locking
#   even after the EMA has drifted - bcs shorts color does not change(much)
# - update_signature: only applies the EMA if the new crop correlates >= 0.7 with
#   the stored sig. If the lock is wrong (looking at wrong person), the correlation
#   will be low and the update is rejected — signature stays clean.
class FighterTracker:
    def __init__(self, fighter_id):
        self.fighter_id = fighter_id
        self.pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.data = []
        self.signature = None       # live EMA signature, slowly adapts
        self.reference_sig = None   # frozen snapshot from initial lock, never updated

    def set_initial_signature(self, sig):
        """Called once at initial lock. Stores both the live and frozen reference copy."""
        self.signature = sig
        self.reference_sig = sig.copy() if sig is not None else None

    def update_signature(self, new_sig):
        """Guarded EMA: only update if new crop looks like what we expect."""
        if new_sig is None or self.signature is None:
            return
        corr = hist_similarity(self.signature, new_sig)
        if corr < SIG_UPDATE_MIN_CORR:
            return  # too different — probably wrong person, don't poison the signature
        self.signature = (1 - SIG_UPDATE_WEIGHT) * self.signature + SIG_UPDATE_WEIGHT * new_sig

    def get_best_signature(self):
        """Returns EMA sig if available, else frozen reference."""
        if self.signature is not None:
            return self.signature
        return self.reference_sig

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
    """Initial lock: pick 2 biggest detections with different track IDs."""
    if len(detections) < 2:
        return None
    top2 = sorted(detections.items(), key=lambda d: box_area(d[1]), reverse=True)[:2]
    # Safety: ensure the two picks are actually different track IDs
    if top2[0][0] == top2[1][0]:
        return None
    # Safety: ensure boxes don't overlap too much (same person with 2 IDs)
    if box_iou(top2[0][1], top2[1][1]) > 0.3:
        return None
    result = []
    for tid, box in top2:
        sig = appearance_signature(img, box)
        result.append((tid, box, sig))
    return result


def relock_by_appearance(trackers, detections, img, frame_width, locked_ids):
    """
    Match fighters to detections with appearance + spatial overlap rejection.
    Only considers fighters whose locked_id is None or missing from detections.
    Rejects ambiguous matches where a detection scores similarly for both fighters.
    """
    if not detections:
        return {}

    # Only try to re-lock fighters who actually need it
    fighters_needing_relock = []
    for fi in range(len(trackers)):
        lid = locked_ids[fi]
        if lid is None or lid not in detections:
            fighters_needing_relock.append(fi)

    if not fighters_needing_relock:
        return {}

    # Precompute signatures for all detections
    det_sigs = {}
    for tid, box in detections.items():
        det_sigs[tid] = appearance_signature(img, box)

    # Exclude detections already used by fighters who are fine
    used_tids = set()
    for fi in range(len(trackers)):
        if fi not in fighters_needing_relock and locked_ids[fi] is not None:
            used_tids.add(locked_ids[fi])

    # Score every (needy fighter, available detection) pair
    # Also collect cross-scores so we can check for ambiguity
    all_scores = {}  # (fi, tid) -> score
    for fi in range(len(trackers)):
        tracker = trackers[fi]
        ema_sig = tracker.signature
        ref_sig = tracker.reference_sig
        if ema_sig is None and ref_sig is None:
            continue
        for tid in detections:
            if tid in used_tids:
                continue
            dsig = det_sigs[tid]
            score_ema = hist_similarity(ema_sig, dsig) if ema_sig is not None else 0.0
            score_ref = hist_similarity(ref_sig, dsig) if ref_sig is not None else 0.0
            all_scores[(fi, tid)] = max(score_ema, score_ref)

    # Build candidate list only for fighters needing re-lock
    candidates = []
    for (fi, tid), score in all_scores.items():
        if fi in fighters_needing_relock:
            candidates.append((score, fi, tid))

    candidates.sort(reverse=True, key=lambda x: x[0])

    assigned_fighters = set()
    assigned_tids = set()
    new_locks = {}
    for score, fi, tid in candidates:
        if score < MIN_MATCH_SCORE:
            break
        if fi in assigned_fighters or tid in assigned_tids:
            continue

        # Ambiguity check: does the OTHER fighter also score well on this detection?
        other_fi = 1 - fi
        other_score = all_scores.get((other_fi, tid), 0.0)
        if other_score > 0 and abs(score - other_score) < 0.1:
            # Both fighters score similarly — this detection is ambiguous, skip it
            continue

        # Spatial check
        new_box = detections[tid]
        too_close = False
        for other_fi_assigned, other_tid in new_locks.items():
            other_box = detections[other_tid]
            if box_iou(new_box, other_box) > 0.3:
                too_close = True
                break
            cx1, _ = box_center(new_box)
            cx2, _ = box_center(other_box)
            if abs(cx1 - cx2) / frame_width < MIN_BOX_CENTER_DIST:
                too_close = True
                break

        if too_close:
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
locked_ids = [None, None]
per_fighter_missing = [0, 0]
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

            pad_x = int((x2 - x1) * 0.15)  # 15% width padding (30% caused constant overlap)
            pad_y = int((y2 - y1) * 0.1)  # 10% height padding

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
                trackers[i].set_initial_signature(sig)  # CHANGED: stores both live + frozen reference
            print(f"[frame {frame_count}] Initial lock: {locked_ids}")

    # --- Per-fighter stale ID detection ---
    for i, lid in enumerate(locked_ids):
        if lid is None:
            per_fighter_missing[i] = 0
        elif lid not in detections:
            per_fighter_missing[i] += 1
            if per_fighter_missing[i] >= MISSING_FRAMES_RELOCK:
                print(f"[frame {frame_count}] Fighter {i} ID {lid} stale for {per_fighter_missing[i]} frames - unlocking")
                locked_ids[i] = None
                per_fighter_missing[i] = 0
        else:
            per_fighter_missing[i] = 0

    # --- Decide whether to re-lock ---
    locked_present = [lid is not None for lid in locked_ids]
    visible = [lid is not None and lid in detections for lid in locked_ids]
    if any(locked_present) and not any(visible):
        missing_streak += 1
    else:
        missing_streak = 0

    need_relock = (
        any(lid is None for lid in locked_ids) or
        scene_cut or
        missing_streak >= MISSING_FRAMES_RELOCK
    )

    if need_relock and frame_count >= INITIAL_LOCK_FRAMES and len(detections) >= 1:
        new_locks = relock_by_appearance(trackers, detections, img, w, locked_ids)
        if new_locks:
            for fi, tid in new_locks.items():
                if locked_ids[fi] != tid:
                    reason = 'scene cut' if scene_cut else ('missing' if missing_streak else 'unlocked')
                    print(f"[frame {frame_count}] Re-lock fighter {fi}: {locked_ids[fi]} -> {tid} ({reason})")
                locked_ids[fi] = tid
            missing_streak = 0

    # --- Same-ID / same-person sanity check ---
    # If both fighters somehow point to the same track ID, that's always wrong.
    # Keep the fighter whose signature matches better, unlock the other.
    if (locked_ids[0] is not None and locked_ids[1] is not None
            and locked_ids[0] == locked_ids[1]):
        # Both on same track ID — compare who matches better
        box = detections.get(locked_ids[0])
        if box is not None:
            dsig = appearance_signature(img, box)
            s0 = hist_similarity(trackers[0].reference_sig, dsig)
            s1 = hist_similarity(trackers[1].reference_sig, dsig)
            loser = 1 if s0 >= s1 else 0
            print(f"[frame {frame_count}] SAME-ID COLLISION: both on {locked_ids[0]}, "
                  f"scores {s0:.2f} vs {s1:.2f}, unlocking fighter {loser}")
            locked_ids[loser] = None

    # If both fighters are locked to different IDs but their boxes overlap heavily,
    # check if they're actually tracking the same person
    if (locked_ids[0] is not None and locked_ids[1] is not None
            and locked_ids[0] != locked_ids[1]
            and locked_ids[0] in detections and locked_ids[1] in detections):
        b0 = detections[locked_ids[0]]
        b1 = detections[locked_ids[1]]
        iou = box_iou(b0, b1)
        if iou > 0.5:  # very high overlap = almost certainly same person
            # Compare reference sigs to decide who's the impostor
            sig0 = appearance_signature(img, b0)
            sig1 = appearance_signature(img, b1)
            ref_score_0 = hist_similarity(trackers[0].reference_sig, sig0)
            ref_score_1 = hist_similarity(trackers[1].reference_sig, sig1)
            loser = 1 if ref_score_0 >= ref_score_1 else 0
            print(f"[frame {frame_count}] HIGH OVERLAP (IoU={iou:.2f}): "
                  f"unlocking fighter {loser}")
            locked_ids[loser] = None

    # --- Process locked fighters (with overlap-aware signature freeze) ---
    active_boxes = {}
    for i, fid in enumerate(locked_ids):
        if fid is not None and fid in detections:
            active_boxes[i] = detections[fid]

    # NEW: if both boxes overlap, freeze signature updates to prevent convergence
    sig_freeze = False
    if len(active_boxes) == 2:
        boxes = list(active_boxes.values())
        iou = box_iou(boxes[0], boxes[1])
        cx0, _ = box_center(boxes[0])
        cx1, _ = box_center(boxes[1])
        center_dist = abs(cx0 - cx1) / w
        if iou > 0.3 or center_dist < MIN_BOX_CENTER_DIST:
            sig_freeze = True

    for i, fid in enumerate(locked_ids):
        if fid is None or fid not in detections:
            continue
        box = detections[fid]
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"Fighter {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        trackers[i].process(img, box, frame_count)
        if not sig_freeze:  # NEW: only update sig when boxes are safely separated
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
