# from ultralytics import YOLO
# import cv2

# model = YOLO("yolo26n.pt")

# results = model.track(
#     source="test_video2.mp4",
#     stream=True,
#     persist=True,
#     tracker="bytetrack.yaml",
#     classes=[0],
#     conf=0.6
# )

# fighter_ids = None
# init_frames = 15
# frame_idx = 0

# for r in results:
#     frame = r.orig_img.copy()
#     h, w = frame.shape[:2]

#     candidates = []

#     if r.boxes is not None:
#         for box in r.boxes:
#             if box.id is None:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             track_id = int(box.id)
#             conf = float(box.conf[0])

#             bw = x2 - x1
#             bh = y2 - y1
#             area = bw * bh
#             bottom_ratio = y2 / h

#             # basic filtering
#             if area < 0.07 * w * h:
#                 continue
#             if bottom_ratio < 0.72:
#                 continue

#             candidates.append({
#                 "id": track_id,
#                 "conf": conf,
#                 "box": (x1, y1, x2, y2),
#                 "area": area
#             })

#     # initialize fighter IDs from first few frames
#     if fighter_ids is None and frame_idx < init_frames:
#         if len(candidates) >= 2:
#             candidates = sorted(candidates, key=lambda d: d["area"], reverse=True)[:2]
#             fighter_ids = {candidates[0]["id"], candidates[1]["id"]}
#             print("Locked fighter IDs:", fighter_ids)

#     # after lock, keep only those two IDs
#     tracked_fighters = []
#     if fighter_ids is not None:
#         tracked_fighters = [c for c in candidates if c["id"] in fighter_ids]

#     for det in tracked_fighters:
#         x1, y1, x2, y2 = det["box"]
#         track_id = det["id"]
#         conf = det["conf"]

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#         cv2.putText(
#             frame,
#             f"fighter id:{track_id} {conf:.2f}",
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2
#         )

#     cv2.imshow("Locked Fighters", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

#     frame_idx += 1

# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import math

VIDEO_PATH = "test_video2.mp4"
MODEL_PATH = "yolo26n.pt"


DISPLAY_SCALE = 0.7
PROCESS_SCALE = 0.5
# How often to run YOLO again to correct tracker drift
DETECTION_EVERY_N_FRAMES = 10

# Minimum confidence for YOLO person detections
YOLO_CONF = 0.6

# Candidate filters
MIN_AREA_RATIO = 0.07      # box area must be at least 7% of frame area
MIN_BOTTOM_RATIO = 0.72    # bottom of box must be low enough in frame

# Matching thresholds
MAX_MATCH_DISTANCE = 180   # pixels
TRACKER_TYPE_PREFERENCE = ["CSRT", "KCF", "MIL"]

def expand_box(box, frame_w, frame_h, fighter_side):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if fighter_side == "left":
        left_pad = 0.20
        right_pad = 0.45
    else:
        left_pad = 0.45
        right_pad = 0.20

    top_pad = 0.12
    bottom_pad = 0.12

    nx1 = max(0, int(x1 - bw * left_pad))
    ny1 = max(0, int(y1 - bh * top_pad))
    nx2 = min(frame_w - 1, int(x2 + bw * right_pad))
    ny2 = min(frame_h - 1, int(y2 + bh * bottom_pad))

    return (nx1, ny1, nx2, ny2)


def create_tracker():
    for name in TRACKER_TYPE_PREFERENCE:
        creator_names = [f"Tracker{name}_create"]
        for creator_name in creator_names:
            if hasattr(cv2, creator_name):
                return getattr(cv2, creator_name)()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, creator_name):
                return getattr(cv2.legacy, creator_name)()
    raise RuntimeError("No suitable OpenCV tracker found. Install opencv-contrib-python.")


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_xyxy(box):
    x, y, bw, bh = box
    return (int(x), int(y), int(x + bw), int(y + bh))


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def dist_centers(box_a, box_b):
    ax, ay = center(box_a)
    bx, by = center(box_b)
    return math.hypot(ax - bx, ay - by)


def area(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = area(box_a) + area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def valid_person_candidate(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    a = area(box)
    bottom_ratio = y2 / float(frame_h)
    if a < MIN_AREA_RATIO * frame_w * frame_h:
        return False
    if bottom_ratio < MIN_BOTTOM_RATIO:
        return False
    return True


def detect_fighter_candidates(model, frame):
    h, w = frame.shape[:2]
    results = model(frame, classes=[0], conf=YOLO_CONF, verbose=False)

    candidates = []
    if not results or results[0].boxes is None:
        return candidates

    for b in results[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0])
        box = clip_box((x1, y1, x2, y2), w, h)

        if not valid_person_candidate(box, w, h):
            continue

        cx, _ = center(box)
        center_penalty = abs(cx - w / 2) / (w / 2)
        score = area(box) - (center_penalty * 0.15 * w * h)

        candidates.append({
            "box": box,
            "conf": conf,
            "score": score
        })

    candidates.sort(key=lambda d: d["score"], reverse=True)
    return candidates


def pick_initial_two_fighters(candidates):
    if len(candidates) < 2:
        return None, None

    top = candidates[:4]
    best_pair = None
    best_pair_score = -1e18

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a = top[i]["box"]
            b = top[j]["box"]

            ax, _ = center(a)
            bx, _ = center(b)

            pair_score = top[i]["score"] + top[j]["score"]

            # Prefer some horizontal separation so we get two fighters, not duplicate-ish boxes
            horizontal_sep = abs(ax - bx)
            pair_score += horizontal_sep * 2.0

            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (top[i], top[j])

    if best_pair is None:
        return None, None

    left, right = sorted(best_pair, key=lambda d: center(d["box"])[0])
    return left["box"], right["box"]


def init_tracker_on_box(frame, box):
    tracker = create_tracker()
    tracker.init(frame, xyxy_to_xywh(box))
    return tracker


def update_tracker(tracker, frame, frame_w, frame_h):
    ok, tracked = tracker.update(frame)
    if not ok:
        return False, None
    box = clip_box(xywh_to_xyxy(tracked), frame_w, frame_h)
    return True, box


def best_match_for_track(track_box, detections, used_indices):
    best_idx = None
    best_score = float("inf")

    for idx, det in enumerate(detections):
        if idx in used_indices:
            continue

        det_box = det["box"]
        d = dist_centers(track_box, det_box)

        if d < best_score:
            best_score = d
            best_idx = idx

    if best_idx is None or best_score > MAX_MATCH_DISTANCE:
        return None, None

    return best_idx, detections[best_idx]["box"]


def draw_box(frame, box, label, color, dashed=False):
    x1, y1, x2, y2 = box

    if not dashed:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    else:
        step = 10
        for x in range(x1, x2, step * 2):
            cv2.line(frame, (x, y1), (min(x + step, x2), y1), color, 2)
            cv2.line(frame, (x, y2), (min(x + step, x2), y2), color, 2)
        for y in range(y1, y2, step * 2):
            cv2.line(frame, (x1, y), (x1, min(y + step, y2)), color, 2)
            cv2.line(frame, (x2, y), (x2, min(y + step, y2)), color, 2)

    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("Could not read first frame.")

frame_h, frame_w = first_frame.shape[:2]

model = YOLO(MODEL_PATH)

initial_candidates = detect_fighter_candidates(model, first_frame)
fighter1_box, fighter2_box = pick_initial_two_fighters(initial_candidates)

if fighter1_box is None or fighter2_box is None:
    raise RuntimeError("Could not find two fighter boxes in the first frame.")

tracker1 = init_tracker_on_box(first_frame, fighter1_box)
tracker2 = init_tracker_on_box(first_frame, fighter2_box)

fighter1_last_box = fighter1_box
fighter2_last_box = fighter2_box

frame_idx = 0

while True:
    if frame_idx == 0:
        frame = first_frame.copy()
        success = True
    else:
        success, frame = cap.read()
        
    if not success:
        break

    
    #frame = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)
    h, w = frame.shape[:2]
    ok1, box11 = update_tracker(tracker1, frame, w, h)
    ok2, box22 = update_tracker(tracker2, frame, w, h)
    box1 = expand_box(box11, w, h, "left")
    box2 = expand_box(box22, w, h, "right")
    if ok1:
        fighter1_last_box = box1
    else:
        box1 = fighter1_last_box

    if ok2:
        fighter2_last_box = box2
    else:
        box2 = fighter2_last_box

    # Every N frames, use YOLO as a correction source.
    # Important: both boxes are always kept, and overlap is allowed.
    if frame_idx % DETECTION_EVERY_N_FRAMES == 0:
        detections = detect_fighter_candidates(model, frame)
        used = set()

        idx1, matched1 = best_match_for_track(box1, detections, used)
        if matched1 is not None:
            box1 = matched1
            fighter1_last_box = matched1
            tracker1 = init_tracker_on_box(frame, matched1)
            used.add(idx1)

        idx2, matched2 = best_match_for_track(box2, detections, used)
        if matched2 is not None:
            box2 = matched2
            fighter2_last_box = matched2
            tracker2 = init_tracker_on_box(frame, matched2)
            used.add(idx2)

    # Draw both boxes always, even if they overlap heavily
    draw_box(frame, box1, "Fighter 1", (0, 255, 0), dashed=not ok1)
    draw_box(frame, box2, "Fighter 2", (255, 0, 0), dashed=not ok2)

    overlap = iou(box1, box2)
    cv2.putText(
        frame,
        f"Overlap IoU: {overlap:.2f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )
    cv2.putText(
        frame,
        "Dashed box = tracker fallback",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

    cv2.imshow("Two Fighters - Overlap Allowed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()




# find a way for box to update sooner - basically as the punch goes
# add mediapipe to that specific box to track body