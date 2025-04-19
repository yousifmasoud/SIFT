import cv2
import numpy as np
import os

# Load target image (used for matching)
target = cv2.imread('input/query2.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT (uses DoG internally)
sift = cv2.SIFT_create()
kp_query, des_query = sift.detectAndCompute(target, None)

# Set up video
cap = cv2.VideoCapture('input/video.mp4')

# FLANN-based matcher for SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(gray, None)

    if des_frame is not None:
        matches = flann.knnMatch(des_query, des_frame, k=2)

        # Lowe's ratio test
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        good = sorted(good, key=lambda x: x.distance)[:30]  # limit to top 30 matches

        if len(good) > 10:
            matched_frame = cv2.drawMatches(target, kp_query, frame, kp_frame, good, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('Object Detection in Video (SIFT/DoG)', matched_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
