"""Example of pykitti.raw usage."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw
import os
# Change this to the directory where you store KITTI data

pleft ='/unsullied/sharefs/jiangying/data/mynteye/left/'
plim = '/unsullied/sharefs/jiangying/data/mynteye/laserpng/left/'

pright = '/unsullied/sharefs/jiangying/data/mynteye/right/'
#生成图片放入以上两个文件夹，图片大小则是看原图
d = 0.12
for pts in os.listdir('/unsullied/sharefs/jiangying/data/mynteye/velo/'):
        img = Image.open(pleft+pts[:-3]+'png')
        R0 = np.array([0.998326,  -0.003448,  0.057728 , 0.002884 , 0.999947 , 0.009852 , -0.057759  ,-0.009669 , 0.998284]).reshape(3,3)
        t = np.zeros((3, 1))
        R0=np.vstack((np.hstack([R0, t]), [0, 0, 0, 1]))

        P2 = np.array([333.730956,  0.000000,  353.883846,  0.000000,  0.000000,  333.730956,  237.043222,  0.000000,  0.000000,  0.000000 , 1.000000 , 0.000000]).reshape(3,4)
        Tr = np.array( [0.000802187,          -1, 3.21752e-07 ,          0,
0.000802187, 3.21752e-07  ,        -1   ,     0.34,
   0.999999, 0.000802187 ,0.000802187    ,   0.183,
          0      ,     0     ,      0     ,      1]).reshape(4,4)
        tvec = [ 0.000, 0.340, 0.183 ]
        rvec = [0.785, -1.570, 0.785 ]
        mtx = np.array(
            [375.081422, 0.000000, 379.639896, 0.000000, 375.519135, 246.490038, 0.000000, 0.000000, 1.000000]).reshape(
            3, 3)
        new_mtx = [[249.81304932 ,  0.     ,    376.72466381],
 [  0.      ,   250.09353638 ,245.90674095],
 [  0.       ,    0.        ,   1.        ]]
        dist = np.array([-0.283679, 0.060266, -0.000654, -0.001146, 0.000000])
        R = np.array(
            [0.998326, -0.003448, 0.057728, 0.002884, 0.999947, 0.009852, -0.057759, -0.009669, 0.998284]).reshape(3, 3)

        # print("pts3d.shape:", pts3d.shape),4
        pts3d = np.load('/unsullied/sharefs/jiangying/data/mynteye/velo/'+pts).T
        pts3d[-1, :] = 1


    # Project 3d points
        '''pts3d_cam = R0 @ Tr @ pts3d
        mask = pts3d_cam[2, :] >= 0  # Z >= 0
        pts2d_cam = P2 @ pts3d_cam[:, mask]

        pts2d = (pts2d_cam / pts2d_cam[2, :])[:-1, :].T
        dpth = (pts2d_cam)[-1, :].T
        '''
        pts3d_cam = (Tr @ pts3d)[:-1,:]
        mask = pts3d_cam[2, :] >= 0
        c,r,t,x,x,x,x = cv2.decomposeProjectionMatrix(P2, mtx)
        assert(c.all() == mtx.all())
        #PP = newmtx @R  @ R和t组装（已经Tr得到）进入相机坐标系
        pts2d_cam = new_mtx @R @ pts3d_cam[:, mask]
        pts2d = (pts2d_cam / pts2d_cam[2, :])[:-1, :].T
        dpth = (pts2d_cam)[-1, :].T
        fx = new_mtx[0][0]
        a = fx*d/dpth

    # Draw the points
        '''
        img_d = Image.new('F', img.size)
        img_draw = ImageDraw.Draw(img_d)
        for j,point in enumerate(pts2d):
            img_draw.point(point, fill=a[j])
        img_d.save(os.path.join(plim,pts[:-3]+'tiff'))
        '''

        amin, amax = a.min(), a.max() # 求最大最小值
        a = (255*(a-amin)/(amax-amin))
        a = np.around(a).astype(np.int32)
        print("pts2d.shape:", pts2d.shape)
        img_draw = ImageDraw.Draw(img)
        for i, point in enumerate(pts2d):
            img_draw.point(point, fill=(a[i], 0, 0))
        img.save('str(idx)' + '.png')
        break
        #print(list(img_d.getdata()))



