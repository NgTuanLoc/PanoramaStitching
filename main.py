from imageWarping import *


images = loadImg(args)
n = len(images)
matching_method = args['method']
feature_extractor = args['descriptor']
ratio = 0.75
# Choose middle image as pivot that dividing and stitching it by left image and right image and so on 


if(n==2):
    fullpano,_,_,_=warpTwoImages(list_images[0],list_images[1])
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

outputFile = 'panoramaImage'
fileName = 'outputs/'+outputFile+'.jpg'

for i in range(100):
    if not os.path.isfile(fileName):
        cv2.imwrite(fileName, fullpano)
        break
    fileName = 'outputs/{}_{}_{}_{}.jpg'.format(outputFile, feature_extractor, matching_method, i)
    # fileName = 'outputs/'+outputFile+str(i)+++'.jpg'