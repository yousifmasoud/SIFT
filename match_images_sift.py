import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
query = cv2.imread('input/query.jpg', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('input/target.jpg', cv2.IMREAD_GRAYSCALE)

# Create SIFT detector with adjusted parameters
sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.04, edgeThreshold=10) #DoG
kp1, des1 = sift.detectAndCompute(query, None)
kp2, des2 = sift.detectAndCompute(target, None)

# FLANN matching setup
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Lowe's Ratio Test (adjusted threshold)
good = [m for m, n in matches if m.distance < 0.7 * n.distance]

MIN_MATCH_COUNT = 10

if len(good) >= MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is not None:
        matches_mask = mask.ravel().tolist()
        inliers = [good[i] for i in range(len(good)) if matches_mask[i]]

        # Spread check — filter out clustered false matches
        dst_inliers = np.float32([kp2[m.trainIdx].pt for m in inliers])
        std_dev = np.std(dst_inliers, axis=0)  # std in x and y

        if np.min(std_dev) > 10:  # adjust threshold if needed
            h, w = query.shape
            box = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            box_transformed = cv2.perspectiveTransform(box, M)

            target_color = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
            cv2.polylines(target_color, [np.int32(box_transformed)], True, (0, 255, 0), 3, cv2.LINE_AA)

            result = cv2.drawMatches(query, kp1, target_color, kp2, inliers, None,
                                     matchesMask=[1]*len(inliers), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.figure(figsize=(16, 12))
            plt.imshow(result)
            plt.title('SIFT Matching with Homography + Spread Filter')
            plt.axis('off')
            plt.savefig('output/result_sift_DoG.png', dpi=600)
            plt.show()
