import cv2
import numpy as np


video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

line_position = 450

vehicle_count = 0
min_contour_width = 40
min_contour_height = 40
offset = 6


def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detects = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    fg_mask = bg_subtractor.apply(frame)

    _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= min_contour_width and h >= min_contour_height:
            center = get_center(x, y, w, h)
            detects.append(center)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y) in detects:
        if (line_position - offset) < y < (line_position + offset):
            vehicle_count += 1
            detects.remove((x, y))
            print(f"Vehicle Count: {vehicle_count}")

    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow('Video', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
