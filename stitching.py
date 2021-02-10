# This module can stitching image but the output is less nice
from featureExtraction import detectAndDescribe
from MatchingFeatures import *
from getHomography import *

def image_stitch(images, matching_method, feature_extractor):
        # Loading images and convert them to gray scale to increase acuracy for detecting features and keypoints step
        (imageB, imageA) = images
        (imageB_gray, imageA_gray) = (cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY))
        
        #detect the features and keypoints 
        (KeypointsA, featuresA) = detectAndDescribe(imageA_gray, feature_extractor)
        (KeypointsB, featuresB) = detectAndDescribe(imageB_gray, feature_extractor)

        #got the valid matched points with knn or brute force matching technique
        
        matches = 0
        if matching_method == 'bf':
            matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        else:
            matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)

        # if there are no keypoints to match => return to fail to panorama these images
        if matches is None:
            print("Can't not match !!!!!")
            return None

        #to get perspective of image using computed homography
        (matches, H, status) = getHomography(KeypointsA, KeypointsB, featuresA, featuresB, matches, reprojThresh=4)
        # Combination of two image shapes
        width = imageA.shape[1] + imageB.shape[1] 
        height = imageA.shape[0] + imageB.shape[0]		
        result_image = cv2.warpPerspective(imageA, H, (width, height))
        result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        return result_image