# SIFT & ORB Image and Video Matching Toolkit

This repository contains Python scripts for feature-based object detection using SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF). These methods are applied to both static image pairs and video streams.

---

## Files

### 1. `match_images.py`  
Detects and matches keypoints between two images using the ORB algorithm and Hamming distance.

- **Algorithm:** ORB  
- **Matcher:** Brute Force with Hamming distance  
- **Output:** `result.png` (top 10 matches displayed)

**Example Output:**

![ORB Matching Result](output/result.png)

---

### 2. `match_images_sift.py`  
Detects keypoints using SIFT (based on Difference of Gaussian). Applies Lowe's ratio test and homography to localize the query image in the target. Filters clustered matches using standard deviation.

- **Algorithm:** SIFT (uses Difference of Gaussian internally)  
- **Matcher:** FLANN with Lowe’s ratio test  
- **Enhancements:**
  - Homography estimation using RANSAC  
  - Spread filtering to remove tightly clustered false positives  
- **Output:** `result_sift_DoG.png` with object bounding box if detection is valid

**Example Output:**

![SIFT Matching Result](output/result_sift_DoG.png)

---

### 3. `detect_in_video.py`  
Uses ORB to detect and track the query object in a video file.

- **Algorithm:** ORB  
- **Matcher:** Brute Force with Hamming distance (cross-check enabled)  
- Displays real-time matches (limited to top 20) between the query and video frames  
- Exit with `ESC` key

---

### 4. `detect_in_video_sift.py`  
Uses SIFT (DoG-based) to match a query image in each frame of a video. Matches are filtered by Lowe's ratio test and sorted by quality.

- **Algorithm:** SIFT  
- **Matcher:** FLANN with Lowe’s ratio test  
- Limits visualized matches to top 30 for clarity  
- Displays real-time detection results  
- Exit with `ESC` key

* Note: If an error occurs when running the file, try running the following command: `LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0 python3 detect_in_video_sift.py`
---

## Requirements

- Python 3.x  
- OpenCV (4.4 or higher recommended)  
- Matplotlib (for static image plotting)

Install dependencies with:

```bash
pip install opencv-python opencv-contrib-python matplotlib
