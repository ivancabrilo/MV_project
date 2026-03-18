

import cv2
import mediapipe as mp
import time
import json
import math

# Initialize video capture and pose detection
cap = cv2.VideoCapture('test_video2.mp4')

if not cap.isOpened():
    raise RuntimeError("Could not open video file: test_video2.mp4")

source_fps = cap.get(cv2.CAP_PROP_FPS)
if source_fps is None or source_fps <= 0:
    source_fps = 30.0

playback_speed_multiplier = 1.75
preview_scale = 0.8

pTime = 0
data = []
fighter_side = input("Track which fighter? (l/r): ").strip().lower()
if fighter_side not in ("l", "r"):
    fighter_side = "l"
side_label = "left" if fighter_side == "l" else "right"
output_json = f"pose_landmarks_{side_label}_fighter.json"

try:
    mpPose = mp.solutions.pose
except AttributeError:
    try:
        from mediapipe.python import solutions as mp_solutions
        mpPose = mp_solutions.pose
    except Exception as e:
        raise RuntimeError(
            "Could not load MediaPipe Pose API. Reinstall mediapipe in the active environment."
        ) from e
pose = mpPose.Pose()


def create_tracker():
    tracker_creators = [
        ("legacy", "TrackerCSRT_create"),
        (None, "TrackerCSRT_create"),
        ("legacy", "TrackerKCF_create"),
        (None, "TrackerKCF_create"),
        ("legacy", "TrackerMIL_create"),
        (None, "TrackerMIL_create"),
    ]

    for namespace, creator_name in tracker_creators:
        if namespace == "legacy" and hasattr(cv2, "legacy") and hasattr(cv2.legacy, creator_name):
            return getattr(cv2.legacy, creator_name)()
        if namespace is None and hasattr(cv2, creator_name):
            return getattr(cv2, creator_name)()

    raise RuntimeError("No OpenCV tracker available. Install opencv-contrib-python.")


def bbox_from_landmarks(results, x_offset, crop_w, full_w, full_h):
    if results is None or not results.pose_landmarks:
        return None

    xs = []
    ys = []
    for lm in results.pose_landmarks.landmark:
        xs.append(((lm.x * crop_w) + x_offset) / full_w)
        ys.append(lm.y)

    min_x = max(0.0, min(xs) - 0.08)
    max_x = min(1.0, max(xs) + 0.08)
    min_y = max(0.0, min(ys) - 0.08)
    max_y = min(1.0, max(ys) + 0.08)

    x = int(min_x * full_w)
    y = int(min_y * full_h)
    bw = max(1, int((max_x - min_x) * full_w))
    bh = max(1, int((max_y - min_y) * full_h))

    x2 = min(full_w, x + bw)
    y2 = min(full_h, y + bh)
    return (x, y, max(1, x2 - x), max(1, y2 - y))


def expand_bbox(x, y, bw, bh, full_w, full_h, pad_ratio=0.45):
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    new_w = bw * (1.0 + 2.0 * pad_ratio)
    new_h = bh * (1.0 + 2.0 * pad_ratio)

    x1 = max(0, int(cx - new_w / 2.0))
    y1 = max(0, int(cy - new_h / 2.0))
    x2 = min(full_w, int(cx + new_w / 2.0))
    y2 = min(full_h, int(cy + new_h / 2.0))

    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def clip_bbox(bbox, frame_w, frame_h):
    x, y, bw, bh = bbox
    x = max(0, int(x))
    y = max(0, int(y))
    bw = max(1, int(bw))
    bh = max(1, int(bh))
    x2 = min(frame_w, x + bw)
    y2 = min(frame_h, y + bh)
    return (x, y, max(1, x2 - x), max(1, y2 - y))


def bbox_center(bbox):
    x, y, bw, bh = bbox
    return (x + bw / 2.0, y + bh / 2.0)


def bbox_center_distance(a, b):
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def compute_roi_hist(frame, bbox):
    x, y, bw, bh = bbox
    patch = frame[y:y + bh, x:x + bw]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def hist_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("Could not read first frame from video.")

first_h, first_w, _ = first_frame.shape
half_w = first_w // 2

if fighter_side == "l":
    side_crop = first_frame[:, :half_w]
    side_offset = 0
    side_width = half_w
else:
    side_crop = first_frame[:, half_w:]
    side_offset = half_w
    side_width = first_w - half_w

side_results = pose.process(cv2.cvtColor(side_crop, cv2.COLOR_BGR2RGB))
roi = bbox_from_landmarks(side_results, side_offset, side_width, first_w, first_h)

if roi is None:
    fallback_w = max(1, int(first_w * 0.35))
    fallback_h = max(1, int(first_h * 0.9))
    fallback_y = int(first_h * 0.05)
    if fighter_side == "l":
        fallback_x = int(first_w * 0.05)
    else:
        fallback_x = max(0, first_w - fallback_w - int(first_w * 0.05))
    roi = (fallback_x, fallback_y, fallback_w, fallback_h)

tracker = create_tracker()
tracker.init(first_frame, roi)

reference_bbox = clip_bbox(roi, first_w, first_h)
reference_hist = compute_roi_hist(first_frame, reference_bbox)
last_good_bbox = reference_bbox
rejected_count = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0
frame_skip = 3  # Process every 3rd frame (higher = faster, less temporal detail)

# List of landmarks to exclude
excluded_landmarks = list(range(0, 11))  # Exclude face landmarks

while True:
    success, img = cap.read()
    if not success:
        break

    # Skip frames to speed up processing
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    h, w, c = img.shape
    tracked, bbox = tracker.update(img)
    active_bbox = None

    if tracked:
        candidate_bbox = clip_bbox(bbox, w, h)
        candidate_hist = compute_roi_hist(img, candidate_bbox)
        appearance_ok = hist_similarity(reference_hist, candidate_hist) >= 0.40

        cx, _ = bbox_center(candidate_bbox)
        side_margin = int(w * 0.08)
        if fighter_side == "l":
            side_ok = cx <= (w // 2) + side_margin
        else:
            side_ok = cx >= (w // 2) - side_margin

        jump_limit = max(80.0, 1.25 * max(last_good_bbox[2], last_good_bbox[3]))
        motion_ok = bbox_center_distance(candidate_bbox, last_good_bbox) <= jump_limit

        if appearance_ok and side_ok and motion_ok:
            active_bbox = candidate_bbox
            last_good_bbox = candidate_bbox
            rejected_count = 0
        else:
            active_bbox = last_good_bbox
            rejected_count += 1
            if rejected_count >= 8:
                tracker = create_tracker()
                tracker.init(img, last_good_bbox)
                rejected_count = 0
                cv2.putText(img, "Re-locking target", (70, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)
    else:
        active_bbox = last_good_bbox
        rejected_count += 1
        if rejected_count >= 8:
            tracker = create_tracker()
            tracker.init(img, last_good_bbox)
            rejected_count = 0
        cv2.putText(img, "Tracker uncertain", (70, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if active_bbox is not None:
        x, y, bw, bh = active_bbox
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 255), 2)

        px, py, pbw, pbh = expand_bbox(x, y, bw, bh, w, h, pad_ratio=0.45)
        px2 = min(w, px + pbw)
        py2 = min(h, py + pbh)
        roi_img = img[py:py2, px:px2]

        cv2.rectangle(img, (px, py), (px2, py2), (255, 200, 0), 1)
    else:
        roi_img = None
        cv2.putText(img, "Tracker lost target", (70, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if roi_img is not None and roi_img.size > 0:
        results = pose.process(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    else:
        results = None

    if results is not None and results.pose_landmarks:
        frame_data = {
            "frame": frame_count,
            "landmarks": []
        }

        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id not in excluded_landmarks:  # Only draw non-face landmarks
                global_x = ((lm.x * pbw) + px) / w
                global_y = ((lm.y * pbh) + py) / h

                cx, cy = int(global_x * w), int(global_y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                if lm.visibility > .65:
                    landmark_data = {
                        "id": id,
                        "x": global_x,
                        "y": global_y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    }
                    frame_data["landmarks"].append(landmark_data)

        data.append(frame_data)

        for connection in mpPose.POSE_CONNECTIONS:
            if connection[0] not in excluded_landmarks and connection[1] not in excluded_landmarks:
                start = results.pose_landmarks.landmark[connection[0]]
                end = results.pose_landmarks.landmark[connection[1]]
                start_global_x = ((start.x * pbw) + px) / w
                start_global_y = ((start.y * pbh) + py) / h
                end_global_x = ((end.x * pbw) + px) / w
                end_global_y = ((end.y * pbh) + py) / h

                start_x, start_y = int(start_global_x * w), int(start_global_y * h)
                end_x, end_y = int(end_global_x * w), int(end_global_y * h)
                cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"Fighter: {side_label}", (70, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (70, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    if preview_scale != 1.0:
        preview_img = cv2.resize(img, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_LINEAR)
    else:
        preview_img = img
    cv2.imshow('Image', preview_img)

    display_fps = (source_fps / frame_skip) * playback_speed_multiplier
    delay = max(1, int(1000 / display_fps))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

    frame_count += 1

# Write JSON data after processing all frames
with open(output_json, 'w') as f:
    json.dump(data, f, indent=4)

cap.release()
cv2.destroyAllWindows()
