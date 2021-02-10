'''
Code Refference : https://github.com/ndvinh98/Web-App-Panorama/blob/master/stitch.py
Author : ndvinh98
Task : These module will make images are warped and stitched much more flexible 
'''

import cv2
import numpy as np
import imutils
from featureExtraction import detectAndDescribe
from MatchingFeatures import matchKeyPointsBF, matchKeyPointsKNN
from getHomography import *
from imagesLoading import loadImg, constructArg
import os


args = constructArg()
matching_method = args['method']
feature_extractor = args['descriptor']

def warpTwoImages(queryImg, trainImg,showstep=False,descriptor='ORB',ratio=0.75):
    '''warp 2 images'''
	#generate Homography matrix
        # Loading images and convert them to gray scale to increase acuracy for detecting features and keypoints step
    (imageB, imageA) = (trainImg, queryImg)
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

	#get height and width of two images
    heightQueryImg,widthQueryImg = imageA.shape[:2]
    heightTrainImg,widthTrainImg = imageB.shape[:2]

	#extract conners of two images: top-left, bottom-left, bottom-right, top-right
    pts1 = np.float32([[0,0],[0,heightQueryImg],[widthQueryImg,heightQueryImg],[widthQueryImg,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,heightTrainImg],[widthTrainImg,heightTrainImg],[widthTrainImg,0]]).reshape(-1,1,2)
    
    
    #apply homography to conners of queryImg
    pts1_ = cv2.perspectiveTransform(pts1, H)
    pts = np.concatenate((pts1_, pts2), axis=0)
    #find max min of x,y coordinate
    [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
    [_, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]

    #top left point of image which apply homography matrix, which has x coordinate < 0, has side=left
    #otherwise side=right (stich to right side of trainImg)
    if(pts[0][0][0]<0): 
        side='left'
        width_pano=widthTrainImg+t[0]
    else:
        width_pano=int(pts1_[3][0][0])
        side='right'
    height_pano=ymax-ymin
    #Translation 
    #https://stackoverflow.com/a/20355545
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 
    queryImg_warped = cv2.warpPerspective(queryImg, Ht.dot(H), (width_pano,height_pano))
    #resize trainImg to the same size as queryImg_warped
    trainImg_rz=np.zeros((height_pano,width_pano,3))
    if side=='left':
        trainImg_rz[t[1]:heightQueryImg+t[1],t[0]:widthTrainImg+t[0]] = trainImg
    else:
        trainImg_rz[t[1]:heightQueryImg+t[1],:widthTrainImg] = trainImg
    #blending panorama, if @display=true, function will return left-side, right-side, of panorama andpanorama w/o blending
    pano,nonblend,leftside,rightside=panoramaBlending(trainImg_rz, queryImg_warped,widthTrainImg,side, showstep=showstep)

    #croping black region
    pano=crop(pano,heightTrainImg,pts)
    return pano,nonblend,leftside,rightside
    

def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    '''create alpha mask.
       @param barrier is x-coordinates of Boundary line between two photos.
       @param smoothing_window is the width of the intersection of two photos.
       @param left_biased=True ->> create left mask, otherwise create right mask
    '''
    assert barrier < width
    mask = np.zeros((height, width))
    
    offset = int(smoothing_window/2)
    try:
        if left_biased:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(1,0,2*offset+1).T, (height, 1))
            mask[:,:barrier-offset] = 1
        else:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(0,1,2*offset+1).T, (height, 1))
            mask[:,barrier+offset:] = 1
    except:
        if left_biased:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(1,0,2*offset).T, (height, 1))
            mask[:,:barrier-offset] = 1
        else:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(0,1,2*offset).T, (height, 1))
            mask[:,barrier+offset:] = 1
    
    return cv2.merge([mask, mask, mask])
    
def panoramaBlending(trainImg_rz,queryImg_warped,widthTrainImg,side,showstep=True):
    '''
    create panorama by adding 2 matrices @trainImg_rz and @queryImg_warped together then blending
    @widthTrainImg is the width of trainImg before resize.
    @side is the direction of queryImg_warped
    '''

    h,w,_=trainImg_rz.shape
    smoothing_window=int(widthTrainImg/8)
    barrier = widthTrainImg -int(smoothing_window/2)
    mask1 = blendingMask(h, w, barrier, smoothing_window = smoothing_window, left_biased = True)
    mask2 = blendingMask(h, w, barrier, smoothing_window = smoothing_window, left_biased = False)

    if showstep:
        nonblend=queryImg_warped+trainImg_rz
    else:
        nonblend=None
        leftside=None
        rightside=None

    if side=='left':
        trainImg_rz=cv2.flip(trainImg_rz,1)
        queryImg_warped=cv2.flip(queryImg_warped,1)
        trainImg_rz=(trainImg_rz*mask1)
        queryImg_warped=(queryImg_warped*mask2)
        pano=queryImg_warped+trainImg_rz
        pano=cv2.flip(pano,1)
        if showstep:
            leftside=cv2.flip(queryImg_warped,1)
            rightside=cv2.flip(trainImg_rz,1)
    else:
        trainImg_rz=(trainImg_rz*mask1)
        queryImg_warped=(queryImg_warped*mask2)
        pano=queryImg_warped+trainImg_rz
        if showstep:
            leftside=trainImg_rz
            rightside=queryImg_warped

    
    return pano,nonblend,leftside,rightside

def crop(panorama,h_dst,conners):
    '''crop panorama based on destination image (trainImg).
    @param panorama is the panorama
    @param h_dst is the hight of destination image
    @param conner is the tuple which containing 4 conners of warped image and 
    4 conners of destination image'''
    #find min of x,y coordinate
    [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
    t = [-xmin,-ymin]
    conners=conners.astype(int)

   #top-left<0 ->>> side = left, otherwise side=right
    if conners[0][0][0]<0:
        n=abs(-conners[1][0][0]+conners[0][0][0])
        panorama=panorama[t[1]:h_dst+t[1],n:,:]
    else:
        if(conners[2][0][0]<conners[3][0][0]):
            panorama=panorama[t[1]:h_dst+t[1],0:conners[2][0][0],:]
        else:
            panorama=panorama[t[1]:h_dst+t[1],0:conners[3][0][0],:]
    return panorama



# Loading Images 
