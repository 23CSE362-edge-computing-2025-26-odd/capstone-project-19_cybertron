import cv2
import torch
import numpy as np
import json
import time
import os

MODEL_ID = "yolov5n"
INPUT = "C:\\Users\\krjee\\OneDrive\\Desktop\\carvid.mp4"
DISPLAY = True
FRAME_SKIP = 4
JSON_PATH = "yolo_vo_state.json"
CLASSES = {"person", "car", "truck", "bus", "motorbike", "bicycle"}
CONF_THRESH = 0.35
NEAR_REL_AREA = 0.015

model = torch.hub.load("ultralytics/yolov5", MODEL_ID, pretrained=True)
cap = cv2.VideoCapture(INPUT)
if isinstance(INPUT, int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

frame_idx = 0
prev_gray = None
prev_kp = None
prev_desc = None
pose = np.eye(4)
poses = [pose.copy()]

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "w") as f:
        json.dump({"records": []}, f)
else:
    with open(JSON_PATH, "w") as f:
        json.dump({"records": []}, f)

def write_record(rec):
    try:
        with open(JSON_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        data = {"records": []}
    data["records"].append(rec)
    if len(data["records"]) > 2000:
        data["records"] = data["records"][-2000:]
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

def bbox_position(x1, x2, W):
    cx = (x1 + x2) / 2.0
    if cx < W/3:
        return "left"
    if cx > 2*W/3:
        return "right"
    return "center"

try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx % FRAME_SKIP != 0:
            if DISPLAY and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        results = model(frame)
        df = results.pandas().xyxy[0]
        H, W = frame.shape[:2]
        frame_area = H * W

        near_detections = []
        for _, r in df.iterrows():
            name = str(r["name"])
            conf = float(r["confidence"])
            if name not in CLASSES or conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])
            box_area = (x2 - x1) * (y2 - y1)
            rel = box_area / frame_area
            is_near = rel >= NEAR_REL_AREA
            pos = bbox_position(x1, x2, W)
            d = {
                "class": name,
                "confidence": round(conf, 3),
                "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                "position": pos,
                "relative_size": round(rel, 4),
                "near": bool(is_near)
            }
            if is_near:
                near_detections.append(d)
            color = (0,255,0) if is_near else (255,255,255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)

        vo_ok = False
        if prev_gray is not None and prev_desc is not None and desc is not None and len(prev_kp) >= 6 and len(kp) >= 6:
            matches = bf.knnMatch(prev_desc, desc, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) >= 8:
                pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in good])
                pts_cur  = np.float32([kp[m.trainIdx].pt for m in good])
                focal = 0.9 * W
                pp = (W/2.0, H/2.0)
                E, mask = cv2.findEssentialMat(pts_cur, pts_prev, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None and E.shape == (3,3):
                    _, R, t, mask_pose = cv2.recoverPose(E, pts_cur, pts_prev, focal=focal, pp=pp)
                    T = np.eye(4)
                    T[:3,:3] = R
                    T[:3,3] = t.ravel()
                    pose = pose @ np.linalg.inv(T)
                    poses.append(pose.copy())
                    vo_ok = True

        prev_gray = gray
        prev_kp = kp
        prev_desc = desc

        tx, ty, tz = float(pose[0,3]), float(pose[1,3]), float(pose[2,3])
        r00, r10 = pose[0,0], pose[1,0]
        yaw = float(np.arctan2(r10, r00))

        rec = {
            "ts": time.time(),
            "frame": frame_idx,
            "pose": {"x": round(tx,3), "y": round(ty,3), "z": round(tz,3), "yaw": round(yaw,3), "vo_ok": bool(vo_ok)},
            "detections": near_detections,
            "obstacle": len(near_detections) > 0
        }

        write_record(rec)

        if DISPLAY:
            cv2.imshow("YOLOv5n + Visual Odometry", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    pass
finally:
    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()
