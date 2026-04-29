import argparse
import json
from pathlib import Path
import cv2


WINDOW_NAME = "Clip Review"
LABEL_KEYS = {
    ord("j"): {"punch": True, "punch_type": "jab"},
    ord("c"): {"punch": True, "punch_type": "cross"},
    ord("h"): {"punch": True, "punch_type": "hook"},
    ord("u"): {"punch": True, "punch_type": "uppercut"},
    ord("n"): {"punch": False, "punch_type": "none"},
    ord("x"): {"punch": True, "punch_type": "unclear"},
}


def load_record(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_record(json_path, record):
    tmp_path = Path(str(json_path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    tmp_path.replace(json_path)


def collect_items(data_root, video_name=None):
    root = Path(data_root)
    if video_name:
        roots = [root / video_name]
    else:
        roots = sorted(path for path in root.iterdir() if path.is_dir())

    items = []
    for video_root in roots:
        if not video_root.exists():
            continue
        for clip_dir in sorted(path for path in video_root.iterdir() if path.is_dir()):
            json_path = clip_dir / "features.json"
            clip_path = clip_dir / "clip.mp4"
            if json_path.exists() and clip_path.exists():
                items.append({
                    "video": video_root.name,
                    "clip_dir": clip_dir,
                    "json_path": json_path,
                    "clip_path": clip_path,
                })
    return items


def is_labeled(record):
    punch_type = record.get("punch_type", "unlabeled")
    punch_value = record.get("punch")
    return punch_type != "unlabeled" or punch_value is False


def format_header(index, total, item, record):
    punch = record.get("punch")
    punch_type = record.get("punch_type", "unlabeled")
    fighter = record.get("fighter", "-")
    arm = record.get("arm", "-")
    peak = record.get("peak_frame", "-")
    return (
        f"[{index + 1}/{total}] {item['video']} / {item['clip_dir'].name}  "
        f"fighter={fighter} arm={arm} peak={peak} punch={punch} type={punch_type}"
    )


def draw_overlay(frame, header, paused):
    lines = [
        header,
        "j=jab | c=cross | h=hook | u=uppercut | n=not punch | x=unclear",
        "space=pause/play | r=replay | b=back | q=quit",
    ]
    if paused:
        lines.append("PAUSED")

    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (frame.shape[1] - 15, 140), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    y = 40
    for line in lines:
        cv2.putText(frame, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
    return frame


def review_clip(item, index, total):
    record = load_record(item["json_path"])
    cap = cv2.VideoCapture(str(item["clip_path"]))
    if not cap.isOpened():
        return "skip"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    delay_ms = max(1, int(1000 / fps))
    paused = False
    clip_finished = False
    last_frame = None

    while True:
        if not paused and not clip_finished:
            ok, frame = cap.read()
            if not ok:
                clip_finished = True
                paused = True
                if last_frame is None:
                    break
                frame = last_frame.copy()
            else:
                last_frame = frame.copy()
        else:
            if last_frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
                last_frame = frame.copy()
            else:
                frame = last_frame.copy()

        header = format_header(index, total, item, record)
        display = draw_overlay(frame.copy(), header, paused)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

        if key in LABEL_KEYS:
            cap.release()
            return LABEL_KEYS[key]
        if key == ord(" "):
            if clip_finished:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                clip_finished = False
                paused = False
                last_frame = None
            else:
                paused = not paused
        elif key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            clip_finished = False
            paused = False
            last_frame = None
        elif key == ord("b"):
            cap.release()
            return "back"
        elif key == ord("q"):
            cap.release()
            return "quit"

    cap.release()
    return "skip"


def apply_label(json_path, label):
    record = load_record(json_path)
    record["punch"] = label["punch"]
    record["punch_type"] = label["punch_type"]
    save_record(json_path, record)


def main():
    parser = argparse.ArgumentParser(
        description="Review saved clip events and write labels directly into each features.json."
    )
    parser.add_argument(
        "--root",
        default="data_collection",
        help="Root directory containing per-video clip folders.",
    )
    parser.add_argument(
        "--video",
        help="Optional video folder inside data_collection, such as test_video2.",
    )
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="Review clips even if they already have a non-unlabeled punch_type.",
    )
    args = parser.parse_args()

    items = collect_items(args.root, args.video)
    if not items:
        print("No clips found.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    index = 0

    while index < len(items):
        record = load_record(items[index]["json_path"])
        if is_labeled(record) and not args.include_labeled:
            index += 1
            continue

        action = review_clip(items[index], index, len(items))
        if isinstance(action, dict):
            apply_label(items[index]["json_path"], action)
            index += 1
        elif action == "back":
            index = max(0, index - 1)
        elif action == "quit":
            break
        else:
            index += 1

    cv2.destroyAllWindows()

    labeled = 0
    for item in items:
        record = load_record(item["json_path"])
        if is_labeled(record):
            labeled += 1
    print(f"Labeled {labeled}/{len(items)} clips under {Path(args.root).resolve()}")


if __name__ == "__main__":
    main()