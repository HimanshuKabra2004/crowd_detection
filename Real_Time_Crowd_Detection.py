import torch
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

# Load the YOLOv5 model from local repo (make sure you've cloned it)
print(" Loading YOLOv5 model...")
model = torch.hub.load('yolov5', 'yolov5s', source='local')
model.conf = 0.4
model.classes = [0]  # detect only 'person'

# Parameters
DISTANCE_THRESHOLD = 75
REQUIRED_FRAMES = 10
CSV_OUTPUT = 'crowd_events_webcam.csv'

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
frame_count = 0
crowd_tracker = defaultdict(int)
logged_events = []

# Helper functions
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def find_crowd_groups(centroids):
    groups = []
    used = set()
    for i, p1 in enumerate(centroids):
        group = [i]
        for j, p2 in enumerate(centroids):
            if i != j and j not in used:
                if euclidean(p1, p2) < DISTANCE_THRESHOLD:
                    group.append(j)
        if len(group) >= 2:
            fgroup = frozenset(group)
            if fgroup not in [frozenset(g) for g in groups]:
                groups.append(group)
            used.update(group)
    return groups

print(" Starting live crowd detection... Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)
    detections = results.xyxy[0]
    centroids = []
    boxes = []

    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        centroids.append((cx, cy))
        boxes.append((x1, y1, x2, y2))

    groups = find_crowd_groups(centroids)

    # Draw detected persons
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Highlight crowd groups
    for group in groups:
        group_id = tuple(sorted(group))
        group_centers = [centroids[i] for i in group]
        group_boxes = [boxes[i] for i in group]

        # Bounding box around group
        x_min = min([b[0] for b in group_boxes])
        y_min = min([b[1] for b in group_boxes])
        x_max = max([b[2] for b in group_boxes])
        y_max = max([b[3] for b in group_boxes])

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        cv2.putText(frame, f"CROWD: {len(group)}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Tracking
        crowd_tracker[group_id] += 1
        if crowd_tracker[group_id] == REQUIRED_FRAMES:
            logged_events.append({
                'Frame Number': frame_count,
                'Person Count in Crowd': len(group)
            })

    # Cleanup stale groups
    current_group_ids = [tuple(sorted(g)) for g in groups]
    for old_group in list(crowd_tracker.keys()):
        if old_group not in current_group_ids:
            del crowd_tracker[old_group]

    # Show frame
    cv2.imshow("Live Crowd Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(" Stopping...")
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(logged_events)
df.to_csv(CSV_OUTPUT, index=False)
print(f"✅ Crowd events saved to: {CSV_OUTPUT}")
