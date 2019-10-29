import nori2 as nori
import time
import  numpy as np
import pickle
import os
import cv2
import re
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


ls = open("./nori_KITTI_train.list", "w")
l = int(-len('2011_10_03_drive_0042_sync/image_02/data/0000000005.png'))
print(l)
with open("realtimelinefirstcspn/csvFile2train_laser_KITTInew.csv", 'r') as f_in:##val
    lines = f_in.readlines()
lines = [x for x in lines if not x.strip()[0] == '#']
nw = nori.open("/unsullied/sharefs/jiangying/data/nori_KITTI_train.nori", "w")
for l in lines:
    to_load = re.split(',|;', l.strip())
    path1 = to_load[0]
    path2 = to_load[1]
    path3 = to_load[2]
    path4 = to_load[3]

    img = cv2.imread(path1)
    # '.jpg'means that the img of the current picture is encoded in jpg format, and the result of encoding in different formats is different.
    #img_encode = cv2.imencode('.jpg', img)[1]
    img_encode = cv2.imencode('.png', img)[1]
    ###data_encode1 = np.array(img_encode)
    data_encode1 = img_encode
    # print(data_encode1)
    img = cv2.imread(path2)
    # '.jpg'means that the img of the current picture is encoded in jpg format, and the result of encoding in different formats is different.
    #img_encode = cv2.imencode('.jpg', img)[1]
    img_encode = cv2.imencode('.png', img)[1]
    ###data_encode2 = np.array(img_encode)
    data_encode2 = img_encode

    img = cv2.imread(path3)
    img_encode = np.array(img)##imencode cannot np.uint16
    ###data_encode2 = np.array(img_encode)
    data_encode3 = img_encode

    # print(data_encode3)
    img = cv2.imread(path4)
    img_encode = cv2.imencode('.png', img)[1]
    ###data_encode2 = np.array(img_encode)
    data_encode4 = img_encode

    path1 = path1[-55:-4]
    ff = {'left':data_encode1,'right':data_encode2,'sgbm':data_encode4,'gt':data_encode3, 'name':path1 }
    #fff = open( "save.p", "wb" )
    filedata = pickle.dumps(ff)

    data_id = nw.put(filedata, filename=path1)

    ls.write("{}\n".format(data_id))
    print(path1+' complete')

ls.close()

nw.close()

