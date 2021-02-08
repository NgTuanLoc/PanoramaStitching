import cv2
import imageio

def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
    
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf', 'brisk', 'orb'"
    # detect and extract features from the image
    descriptor= cv2.xfeatures2d.SIFT_create()
    if method == 'sift':
        descriptor= cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)


