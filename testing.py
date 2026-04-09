import cv2 
import mediapipe as mp
import json
from ultralytics import YOLO


VIDEO_PATH = "test_video3.mp4"
MODEL_PATH = "yolo26n.pt"


excluded_landmarks = list(range(0, 11)) + list(range(25,33)) # ones we need
mpPose = mp.solutions.pose # making an instance of mediapipe api

class FighterTracker:
    def __init__(self, fighter_id):
        self.fighter_id = fighter_id
        self.pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.data = []

    
    def process_frame(self, img, box, frame_count):
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2] # interesting to see if increase in box size does anything - buffer for better accuracy potentially
        if crop.size == 0:
            return 
        ch, cw = crop.shape[:2]

        frame_results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not frame_results.pose_landmarks: # corrupted frame, see what can be done if this occurs, will it lose track of a fighter - usually it does
            # maybe make it so that it re-captures the lost fighter, how?
            return
        
        frame_data = {"frame": frame_count, "fighter": self.fighter_id, "landmarks": []}
        

        for id, lm in enumerate(frame_results.pose_landmarks.landmark):
            if id not in excluded_landmarks:
                cx = int(lm.x * cw) + x1 # to center the coordiantes in the box
                cy = int(lm.y * ch) + y1 # -||- as above

                color = (255,0,0) if self.fighter_id == 0 else (0,0,255)
                cv2.circle(img, (cx,cy), 5, color, cv2.FILLED)


                if lm.visibility > 0.35:
                    frame_data["landmarks"].append({
                        "id": id,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })

        # connecting points with lines
        for connection in mpPose.POSE_CONNECTIONS:
            if connection[0] not in excluded_landmarks and connection[1] not in excluded_landmarks:
                start = frame_results.pose_landmarks.landmark[connection[0]]
                end = frame_results.pose_landmarks.landmark[connection[1]]
                sx = int(start.x * cw) + x1
                sy = int(start.y * ch) + y1
                ex = int(end.x * cw) + x1
                ey = int(end.y * ch) + y1
                color = (0, 255, 0) if self.fighter_id == 0 else (0, 165, 255)
                cv2.line(img, (sx, sy), (ex, ey), color, 2)

        self.data.append(frame_data)


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2-x1) * (y2-y1)



# Custom bytetrack config — more conservative ID assignment
# Save this as custom_bytetrack.yaml in your project folder
BYTETRACK_CONFIG = """
tracker_type: bytetrack
track_high_thresh: 0.6      # only high confidence detections start new tracks
track_low_thresh: 0.1       # low confidence detections can continue existing tracks
new_track_thresh: 0.7       # even higher bar to create a brand new ID
track_buffer: 60            # remember a lost ID for 60 frames before dropping it
match_thresh: 0.7           # how similar boxes need to be to match (higher = stricter)
fuse_score: True
"""


import os
with open("custom_bytetrack.yaml", "w") as f:
    f.write(BYTETRACK_CONFIG)

cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(MODEL_PATH)
video_fps = cap.get(cv2.CAP_PROP_FPS)

trackers = [FighterTracker(0), FighterTracker(1)]
locked_ids = None
frame_count = 0


while True:
    success, img = cap.read()
    if not success:
        break # can we do sth here?

    h, w = img.shape[:2]

    results = model.track(
        img,
        classes=[0],
        conf=0.5,
        persist=True,
        verbose=False,
        tracker="custom_bytetrack.yaml"
    )

    detections = {}

    if results[0].boxes is not None and results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box)
            tid = int(track_id)
            if box_area((x1, y1, x2, y2)) < 0.07 * w * h:
                continue
            detections[tid] = (x1, y1, x2, y2)

    # Lock onto 2 largest people in first 20 frames
    if locked_ids is None and frame_count < 20:
        if len(detections) >= 2:
            top2 = sorted(detections.items(), key=lambda d: box_area(d[1]), reverse=True)[:2]
            locked_ids = [top2[0][0], top2[1][0]]
            print(f"Locked fighter IDs: {locked_ids}")

    if locked_ids:
        for i, fid in enumerate(locked_ids):
            if fid not in detections:
                # Fighter not visible this frame — skip, no fallback
                continue

            box = detections[fid]
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"Fighter {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            trackers[i].process_frame(img, box, frame_count)

    cv2.imshow("Fighters", img)
    if cv2.waitKey(max(1, int(1000 / video_fps))) & 0xFF == ord('q'):
        break

    frame_count += 1

for tracker in trackers:
    with open(f'landmarks_fighter_{tracker.fighter_id}.json', 'w') as f:
        json.dump(tracker.data, f, indent=4)

cap.release()
cv2.destroyAllWindows()