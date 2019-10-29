import nori2 as nori
import time
import numpy as np
import pickle
import os
import cv2

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
    sgbm = cv2.imdecode(np.frombuffer(r['sgbm'], dtype=np.uint8), cv2.IMREAD_COLOR)
    name = r['name']
    cv2.imwrite('temp1.png', left_img)

    print(mat)

    print(r['gt'])

end = time.time()

print('Finishing reading nori in {}s'.format(end - start))
