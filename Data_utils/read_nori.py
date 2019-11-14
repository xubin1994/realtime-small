import nori2 as nori
import time
import numpy as np
import pickle
import os
import cv2
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

start = time.time()

nross = nori.Fetcher()
f = nori.utils.smart_open("s3://bucketjy/nori_sample_dir.nori.list", "r")

lines = f.read().splitlines()  # 读取全部内容
for data_id in lines:
    print('data_id : {}({})'.format(data_id, type(data_id)))

    r = pickle.loads(nross.get(data_id))
    mat = np.frombuffer(r['left'], dtype=np.uint8)
    left_img = cv2.imdecode(mat, cv2.IMREAD_COLOR)  # 540 * 960
    right_img = cv2.imdecode(np.frombuffer(r['right'], dtype=np.uint8), cv2.IMREAD_COLOR)
    left_disp = r['gt']###or left_disp = cv2.imdecode(np.frombuffer(r['gt'], dtype=np.uint16), cv2.IMREAD_GRAYSCALE)


    Sgbm = SGBM(longedge=max(left_img.shape[0], left_img.shape[1]), downscale=1.0, wsize=5, minDisp=0,
                maxDisparity=256)
    sgbm = Sgbm.solve(left_img, right_img)

    # sgbm = sgbm[:, 160:]#黑边 done
    # print(sgbm.shape)
    # sgbm = cv2.resize(sgbm, (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    sgbm = np.expand_dims(sgbm, axis=-1)
    # print(sgbm.shape)



    name = r['name']
    cv2.imwrite('temp1.png', left_img)

    print(mat)

    print(r['gt'])

end = time.time()

print('Finishing reading nori in {}s'.format(end - start))
