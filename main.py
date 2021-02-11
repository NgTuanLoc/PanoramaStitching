from imageWarping import *
from stitching import image_stitch
import matplotlib.pyplot as plt

images = loadImg(args)
n = len(images)
matching_method = args['method']
feature_extractor = args['descriptor']
ratio = 0.75
choice = int(input("Do you want to use warp module (this will make output much prettier) press 0 to deny and 1 to accept : "))

# method 1
if choice == 0:
    for i in range(n):
        images[i] = imutils.resize(images[i], height=400)

    if n==2:
        fullpano = image_stitch(images, matching_method, feature_extractor)
    else:
        l = [images[n-2], images[n-1]]
        print(len(l))
        fullpano = image_stitch([images[n-2], images[n-1]], matching_method, feature_extractor)
        for i in range(n - 2):
            fullpano = image_stitch([images[n-i-3],fullpano], matching_method, feature_extractor)


    # transform the panorama image to grayscale and threshold it 
    gray = cv2.cvtColor(fullpano, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

    # crop the image to the bbox coordinates
    fullpano = fullpano[y:y + h, x:x + w]


    outputFile = 'panoramaImage'
    fileName = 'outputs/'+outputFile+'.jpg'
    for i in range(100):
        if not os.path.isfile(fileName):
            cv2.imwrite(fileName, fullpano)
            break
        fileName = 'outputs/'+outputFile+str(i)+'.jpg'
        
    plt.figure(figsize=(20,10))
    plt.imshow(fullpano)
    plt.axis('off')
    plt.show()
#method 2
elif choice == 1:
    # Choose middle image as pivot that dividing and stitching it by left image and right image and so on 

    #method 2
    if(n==2):
        fullpano,_,_,_=warpTwoImages(images[0],images[1])
    elif (n>2):
        mid=int(n/2+0.5)
        left=images[:mid]   # List of first image to middle image
        right=images[mid-1:] # List of middle image to the last
        right.reverse()
        while len(left)>1:
            trainImg=left.pop() # To keep the left to right rule when panoramating
            queryImg=left.pop() #
            left_pano,_,_,_=warpTwoImages(queryImg,trainImg,descriptor=feature_extractor, ratio=ratio)
            left_pano=left_pano.astype('uint8')
            left.append(left_pano)
        while len(right)>1:
            trainImg=right.pop() # To keep the left to right rule when panoramating
            queryImg=right.pop() #
            right_pano,_,_,_=warpTwoImages(queryImg,trainImg,descriptor=feature_extractor, ratio=ratio)
            right_pano=right_pano.astype('uint8')
            right.append(right_pano)
        #if width_right_pano > width_left_pano, Select right_pano as destination.Otherwise is left_pano => so that the small image (train Image) will 'follow' the big image (Query Img) to stitch 
        
        if(right_pano.shape[1]>=left_pano.shape[1]):
            fullpano,_,_,_=warpTwoImages(left_pano,right_pano,descriptor=feature_extractor, ratio=ratio)
        else:
            fullpano,_,_,_=warpTwoImages(right_pano,left_pano,descriptor=feature_extractor, ratio=ratio)

    outputFile = '{}_{}_{}'.format(args['images'][6:], feature_extractor, matching_method)
    for i in range(100):
        fileName = 'outputs/'+outputFile+'.jpg'
        if not os.path.isfile(fileName):
            cv2.imwrite(fileName, fullpano)
            break
        fileName = 'outputs/'+outputFile+str(i)+'.jpg'