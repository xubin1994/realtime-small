import nori2 as nori
import time
import  numpy as np
import pickle
import os
import cv2
import PIL.Image as Image
import re

class SGBM():
    def __init__(self, longedge=960, wsize=5, downscale=1.0, minDisp=0, maxDisparity=256):

        self.downScale = downscale
        self.longEdge = int(longedge / self.downScale)
        self.minDisp = int(minDisp / self.downScale)
        self.maxDisparity = int(maxDisparity / self.downScale)

        if self.maxDisparity % 16 != 0:
            self.maxDisparity += 16 - self.maxDisparity % 16

        self.wsize = max(1, int(wsize / self.downScale))

        if self.wsize % 2 == 0:
            self.wsize += 1

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.minDisp,
            numDisparities=self.maxDisparity,
            blockSize=self.wsize,
            P1=8 * 3 * self.wsize * self.wsize,
            P2=64 * 3 * self.wsize * self.wsize,
            disp12MaxDiff=-1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=1,  # 32,
            preFilterCap=10,  # 63,
            mode=cv2.STEREO_SGBM_MODE_HH4,
        )

    def solve(self, left, right):

        h, w = left.shape[:2]##注意是反
        # jy--
        # if left.shape[1] > left.shape[0]:
        #     outw = self.longEdge
        #     outh = int(outw / left.shape[1] * left.shape[0])
        # else:
        #     outh = self.longEdge
        #     outw = int(outh / left.shape[0] * left.shape[1])
        #
        # left = cv2.resize(left, (outw, outh), interpolation=cv2.INTER_AREA)
        # right = cv2.resize(right, (outw, outh), interpolation=cv2.INTER_AREA)

        left = cv2.copyMakeBorder(left, 0, 0, self.maxDisparity, 0, cv2.BORDER_REPLICATE)
        right = cv2.copyMakeBorder(right, 0, 0, self.maxDisparity, 0, cv2.BORDER_REPLICATE)

        disp = self.stereo.compute(left, right)

        disp = disp.astype('float32') / 16.0 * self.downScale
        # disp = cv2.convertScaleAbs(disp, alpha=1.0 / 16.0 * self.downScale, beta=-self.orgMinDisp * 4)
        disp = disp[:, self.maxDisparity:]

        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_NEAREST)
        return disp

def readPFM(file):
    """
    Load a pfm file as a numpy array
    Args:
        file: path to the file to be loaded
    Returns:
        content of the file as a numpy array
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dims = file.readline()
    try:
        width, height = list(map(int, dims.split()))
    except:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


# path1 = "/unsullied/sharefs/jiangying/data/flying/frames/frames_cleanpass/TRAIN/A/0000/left/0006.png"  # 文件夹目录
# path2 = "/unsullied/sharefs/jiangying/data/flying/frames/frames_cleanpass/TRAIN/A/0000/right/0006.png"
# path3  = "/unsullied/sharefs/jiangying/data/flying/disparity/TRAIN/A/0000/left/0006.pfm"


ls = open("./nori2_flying_train.list", "w")
# l = int(-len('2011_10_03_drive_0042_sync/image_02/data/0000000005.png'))
# print(l)
# with open("realtimelinefirstcspn/csvFile2train_laser_KITTInew.csv", 'r') as f_in:##val
#     lines = f_in.readlines()


# lines = [x for x in lines if not x.strip()[0] == '#']
nw = nori.open("/data/flying/nori2_flying_train.nori", "w")


nross = nori.Fetcher()
f = nori.utils.smart_open("s3://bucketjy/nori_flying_train.list", "r")
lines = f.read().splitlines()
for data_id in lines:
    print('data_id : {}({})'.format(data_id, type(data_id)))

    r = pickle.loads(nross.get(data_id))
    mat = np.frombuffer(r['left'], dtype=np.uint8)
    left_img = cv2.imdecode(mat, cv2.IMREAD_COLOR)  # 540 * 960
    right_img = cv2.imdecode(np.frombuffer(r['right'], dtype=np.uint8), cv2.IMREAD_COLOR)
    left_disp = r['gt']  ###or left_disp = cv2.imdecode(np.frombuffer(r['gt'], dtype=np.uint16), cv2.IMREAD_GRAYSCALE)

    Sgbm = SGBM(longedge=max(left_img.shape[0], left_img.shape[1]), downscale=1.0, wsize=5, minDisp=0,
                maxDisparity=256)
    sgbm = Sgbm.solve(left_img, right_img)

    # sgbm = sgbm[:, 160:]#黑边 done
    # print(sgbm.shape)
    # sgbm = cv2.resize(sgbm, (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    sgbm = np.expand_dims(sgbm, axis=-1)
    # print(sgbm.shape)

    r['sgbm']=sgbm


    ff = {'left':r['left'],'right':r['right'],'sgbm':sgbm,'gt':r['gt']}#, 'name':path1 }
    #fff = open( "save.p", "wb" )
    filedata = pickle.dumps(ff)

    data_id = nw.put(filedata, filename=data_id)

    ls.write("{}\n".format(data_id))
    print(data_id+' complete')

ls.close()

nw.close()

