import cv2

# Load query image
query = cv2.imread('query2.jpg', cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
kp_query, des_query = orb.detectAndCompute(query, None)

# Set up video
cap = cv2.VideoCapture('video.mp4')

# Use Hamming distance with cross-check for ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    if des_frame is not None:
        matches = bf.match(des_query, des_frame)
        good_matches = sorted(matches, key=lambda x: x.distance)[:20]  # top 20 matches

        # Draw matches
        matched_frame = cv2.drawMatches(query, kp_query, frame, kp_frame, good_matches, None, flags=2)
        cv2.imshow('Object Detection in Video (ORB)', matched_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
