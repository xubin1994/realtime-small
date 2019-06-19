##laser->cam
import cv2
import numpy as np
import os
pleft ='/unsullied/sharefs/_research_slam/data/mynteye/mynteye/left/'
pright = '/unsullied/sharefs/_research_slam/data/mynteye/mynteye/right/'
mtx = np.array([375.081422 , 0.000000 , 379.639896 , 0.000000 , 375.519135 , 246.490038 , 0.000000 , 0.000000 , 1.000000]).reshape(3,3)
dist = np.array([-0.283679,  0.060266 , -0.000654 , -0.001146 , 0.000000])
R =np.array([0.998326  ,-0.003448 , 0.057728 , 0.002884 , 0.999947,  0.009852 , -0.057759 , -0.009669 , 0.998284]).reshape(3,3)

for img in os.listdir(pleft):
    im = cv2.imread(pleft+img)

    h, w = im.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
    if (img.startswith('000000')):
        print(newcameramtx)
    #dst = cv2.undistort(im, mtx, dist, None, newcameramtx)
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, R, newcameramtx, (w, h), cv2.CV_32FC1)
    dst = cv2.remap(im, mapx, mapy, cv2.INTER_CUBIC)
    x, y, w, h = roi
    dst2 = dst[y:y + h, x:x + w]
    cv2.imwrite('/unsullied/sharefs/jiangying/data/mynteye/left/'+img[:-3]+'png',dst2)
mtx = np.array([368.614764,  0.000000,  353.605431,  0.000000,  370.166046,  226.418560 , 0.000000,  0.000000,  1.000000]).reshape(3,3)
dist = np.array([-0.275444,  0.059009 , 0.000778,  0.002175 , 0.000000])
R =np.array([0.999915,  -0.008860,  -0.009599,  0.008766 , 0.999913 , -0.009807,  0.009685,  0.009722,  0.999906]).reshape(3,3)
for img in os.listdir(pright):
    im = cv2.imread(pright+img)
    h, w = im.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
    if(img.startswith('000000')):
        print(newcameramtx)
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, R, newcameramtx, (w, h), cv2.CV_32FC1)
    dst = cv2.remap(im, mapx, mapy, cv2.INTER_CUBIC)
    x, y, w, h = roi
    dst2 = dst[y:y + h, x:x + w]
    cv2.imwrite('/unsullied/sharefs/jiangying/data/mynteye/right/'+img[:-3]+'png',dst2)


