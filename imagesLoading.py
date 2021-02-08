# import library
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import imageio
 
# loading image files and choosing descriptor
# construct the argument parser and parse the arguments
def constructArg():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", type=str, required=True,
        help="path to input directory of images to stitch")
    ap.add_argument("-d", "--descriptor", type=str, required=False,
        help="descriptor", default='sift')
    ap.add_argument("-m", "--method", type=str, required=False,
        help="matching method", default='knn')
    return vars(ap.parse_args())


def loadImg(args):
    # grab the paths to the input images and initialize our images list
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(args["images"])))
    n_of_images = len(imagePaths)
    images = []
    flag = str(input("Do you want to sort images by yourself or they will be sorted alphabetically ? (yes or no):  ")).upper()
    
    if flag == "YES":
        imagePaths = []
        for i in range(n_of_images):
            print("Enter the %d image:" %(i+1))
            imagePaths.append(args["images"]+"/"+input())
    
    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    for imagePath in imagePaths:
        print(imagePath)
        image = cv2.imread(imagePath)
        images.append(image)
    return images