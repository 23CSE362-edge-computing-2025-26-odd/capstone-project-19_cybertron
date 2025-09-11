import cv2
import numpy as np

def detect_lanes(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Focus only on the bottom half of the frame (where lanes are)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, 
                            minLineLength=100, maxLineGap=50)

    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Overlay detected lines on original frame
    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return combined

cap = cv2.VideoCapture("C:/Users/chsai/OneDrive/Desktop/review_2_lane_detection_with_yolo_slam/4K Road traffic video for object detection and tracking - free download now.mp4")  # put your video file here

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lane_frame = detect_lanes(frame)

    cv2.imshow("Lane Detection", lane_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
