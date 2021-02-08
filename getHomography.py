import cv2
import numpy as np
def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) >= 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches]).reshape(-1,1,2)
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches]).reshape(-1,1,2)
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    else:
        print("Can't not match !")
        return None