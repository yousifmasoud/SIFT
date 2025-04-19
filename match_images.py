import cv2
import matplotlib.pyplot as plt

# Load images
query = cv2.imread('query.jpg', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('target.jpg', cv2.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv2.ORB_create()
 
# determine keypoints and descriptors by using the ORB algorithm
 
kpQuery, descQuery = orb.detectAndCompute(query,None)
kpScene, descScene = orb.detectAndCompute(target,None)

# we use the Hamming-distance matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
detectedMatches = bf.match(descQuery,descScene)
 
# sort the detected matches
sortedDetectedMatches = sorted(detectedMatches, key = lambda x:x.distance)
 
# Show first 10 matches.
combinedImage = cv2.drawMatches(query,kpQuery,
                       target,kpScene,
                       sortedDetectedMatches[:10],None,flags=2)
 
plt.figure(figsize=(16,16))
plt.imshow(combinedImage)
plt.savefig('result.png',dpi=600)
plt.show()
