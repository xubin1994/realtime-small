import tensorflow as tf
import numpy as np
import sys
import cv2
import re
import os
from collections import defaultdict
import nori2 as nori
import cv2
from Data_utils import preprocessing
import pickle
import nori2 as nori
nross = nori.Fetcher()

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

def read_pickle(data_id):
    nross = nori.Fetcher()
    data_id = data_id.tolist()[0].decode("utf-8") ##!!!!
    # print_op = tf.print(str(data_id)+"hello", output_stream=sys.stdout)
    # with tf.control_dependencies([print_op]):
    r = pickle.loads(nross.get(data_id))
    # print(r)
    # print("hello")
    mat = np.frombuffer(r['left'], dtype=np.uint8)
    left_img = cv2.imdecode(mat, cv2.IMREAD_COLOR)  # 540 * 960
    # left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.imdecode(np.frombuffer(r['right'], dtype=np.uint8), cv2.IMREAD_COLOR)#np.array
    # right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    #nross.close() no attr
    left_disp = r['gt']
    if 'name' in r.keys():
        name = r['name']
    else:
        name = ''
    if 'sgbm' in r.keys():

        # sgbm = cv2.imdecode(np.frombuffer(r['sgbm'], dtype=np.uint8), cv2.IMREAD_COLOR)#gray也是这么大但是一个通道，错的
        # print(sgbm.shape)

        Sgbm = SGBM(longedge=max(left_img.shape[0], left_img.shape[1]), downscale=1.0, wsize=5, minDisp=0,
                    maxDisparity=256)
        sgbm = Sgbm.solve(left_img, right_img)

        # sgbm = sgbm[:, 160:]#黑边 done
        # print(sgbm.shape)
        # sgbm = cv2.resize(sgbm, (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        sgbm=np.expand_dims(sgbm,axis=-1)
        #print(sgbm.shape)

    else:
        sgbm = None


    #bgr_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #image = np.transpose(image, (0, 2, 3, 1))这俩都不用

    return left_img, right_img, left_disp, sgbm, name



def read_list_file(path_file):
    """
    Read dataset description file encoded as left;right;disp;conf
    Args:
        path_file: path to the file encoding the database
    Returns:
        [left,right,gt,conf] 4 list containing the images to be loaded
    """
    #with open(path_file, 'r') as f:
    f = nori.utils.smart_open(path_file, "r")

    lines = f.read().splitlines()

    pk_file_list = list(lines)
    return pk_file_list


def read_image_from_disc(image_path, shape=None, dtype=tf.uint8):
    """
    Create a queue to hoold the paths of files to be loaded, then create meta op to read and decode image
    Args:
        image_path: metaop with path of the image to be loaded
        shape: optional shape for the image
    Returns:
        meta_op with image_data
    """
    image_raw = tf.read_file(image_path)##原顺序RGB
    if dtype == tf.uint8:
        image = tf.image.decode_image(image_raw)#RGB
    else:
        image = tf.image.decode_png(image_raw, dtype=dtype)
    if shape is None or shape[0] is None:
        image.set_shape([None, None, 3])
    else:
        #image.set_shape(shape)
        image=tf.image.resize_image_with_crop_or_pad(image, shape[0], shape[1])
        image.set_shape(shape)
    return tf.cast(image, dtype=tf.float32)  ###???成了tensor


class dataset():
    """
    Class that reads a dataset for deep stereo
    """

    def __init__(
            self,
            path_file,
            batch_size=4,
            crop_shape=[320, 960],
            num_epochs=None,
            augment=False,
            is_training=True,
            shuffle=True):

        # if not os.path.exists(path_file):
        #     raise Exception('File not found during dataset construction')

        self._path_file = path_file
        self._batch_size = batch_size
        self._crop_shape = crop_shape
        self._num_epochs = num_epochs
        self._augment = augment
        self._shuffle = shuffle
        self._sgbm = False
        self._is_training = is_training
        self._double_prec_las = False
        self._build_input_pipeline()

    def _load_image(self, data_id):

        left_image, right_image, gt_image, sgbm_image,name = tf.py_func(lambda x: read_pickle(x), [data_id], [tf.uint8, tf.uint8,tf.float32, tf.float32, tf.string])##name自己消化
        # origin_shape = [None, None, 1]

        # left_file_name = files[0]
        # right_file_name = files[1]
        # gt_file_name = files[2]
        # if (files.shape[0] > 3):
        #     laser_file_name = files[3]
        if self._usePfm:
            gt_image.set_shape([None, None, 1])
            self._double_prec_las = False
        else:
            self._double_prec_gt = True
            origin_shape=[374, 1238, 1]
            # read_type = tf.uint16 if self._double_prec_gt else tf.uint8
            gt_image = tf.image.resize_image_with_crop_or_pad(gt_image, 374, 1238)##可以输入tensor。。
            left_image = tf.image.resize_image_with_crop_or_pad(left_image, 374, 1238)
            right_image = tf.image.resize_image_with_crop_or_pad(right_image, 374, 1238)
            gt_image = tf.cast(gt_image, tf.float32)
            # if read_type == tf.uint16:#####不是即可
            if self._double_prec_gt: #这里一定是
                gt_image = gt_image / 256.0
        if sgbm_image != None:
            read_type = tf.uint16 if self._double_prec_las else tf.uint8
            #不能读取 sgbm_image = tf.image.resize_image_with_crop_or_pad(sgbm_image,gt_image.get_shape().as_list()[1], gt_image.get_shape().as_list()[2])
            sgbm_image = tf.cast(sgbm_image, tf.float32)
            # sgbm_image = tf.where(tf.greater(sgbm_image, 254.5), sgbm_image,
            #                        tf.zeros_like(sgbm_image, dtype=tf.float32))

            # if self._double_prec_las:  #####不是即可
            #     laser_image = laser_image / 256.0
            sgbm_image =sgbm_image / 16.0


        if self._usePfm:##same as gt
            line_image = gt_image
            line_image.set_shape([540, 960, 1])
            a = line_image.get_shape().as_list()[0]
            b = line_image.get_shape().as_list()[1]
            c = line_image.get_shape().as_list()[2]
            part1 = tf.zeros([270, b, c])
            part2 = tf.zeros([a - 272, b, c])
            val = tf.cast(line_image[270:272], tf.float32)
            line_image = tf.concat([part1, val, part2], axis=0)
        else:
            g = name
            b1 = tf.cond(tf.strings.regex_full_match(g, tf.convert_to_tensor('.*09_26.*')),
                         lambda: tf.constant(1.0),
                         lambda: tf.constant(0.0))
            b2 = tf.cond(tf.strings.regex_full_match(g, tf.convert_to_tensor('.*09_28.*')),
                         lambda: tf.constant(1.0),
                         lambda: tf.constant(0.0))
            b3 = tf.cond(tf.strings.regex_full_match(g, tf.convert_to_tensor('.*09_30.*')),
                         lambda: tf.constant(1.0),
                         lambda: tf.constant(0.0))
            b4 = tf.cond(tf.strings.regex_full_match(g, tf.convert_to_tensor('.*10_03.*')),
                         lambda: tf.constant(1.0),
                         lambda: tf.constant(0.0))
            b5 = tf.cond(tf.strings.regex_full_match(g, tf.convert_to_tensor('.*09_29.*')),
                         lambda: tf.constant(1.0),
                         lambda: tf.constant(0.0))
            fx = b1 * tf.constant(7.215377e+02) + b2 * tf.constant(7.070493e+02) + b3 * tf.constant(
                7.070912e+02) + b4 * tf.constant(7.188560e+02) + b5 * tf.constant(7.183351e+02)
            mask = gt_image > 0
            mask = tf.where(tf.equal(mask, False), tf.zeros_like(gt_image, dtype=tf.float32),
                            tf.ones_like(gt_image, dtype=tf.float32))
            gt_image = fx * 0.54 / (gt_image + 1.0 - mask)  ##unvalid处取1
            gt_image *= mask
            #gt_image = gt_image[:, :tf.shape(left_image)[1], :]


            #read_type = tf.uint16 if self._double_prec_las else tf.uint8
            #line_image = read_image_from_disc(gt_file_name, shape=[None, None, 1], dtype=read_type)
            line_image = gt_image#tf.cast(line_image, tf.float32)
            part1 = tf.zeros([270, line_image.get_shape().as_list()[1], line_image.get_shape().as_list()[2]])
            part2 = tf.zeros([line_image.get_shape().as_list()[0] - 272, line_image.get_shape().as_list()[1],
                              line_image.get_shape().as_list()[2]])
            val = tf.cast(line_image[270:272], tf.float32)
            line_image = tf.concat([part1, val, part2], axis=0)
            #if self._double_prec_las:  #####不是即可
                #line_image = line_image / 256.0




        # crop gt to fit with image (SGM add some paddings who know why...)
        gt_image = gt_image[:, :tf.shape(left_image)[1], :]
        line_image = line_image[:, :tf.shape(left_image)[1], :]

        if sgbm_image != None:

            laser_image = sgbm_image[:, :tf.shape(left_image)[1], :]

        if self._is_training:
            if sgbm_image != None:
                left_image, right_image, gt_image, laser_image,line_image = preprocessing.random_crop(self._crop_shape,
                                                                                           [left_image, right_image,
                                                                                            gt_image, laser_image,line_image])
            else:
                left_image, right_image, gt_image ,line_image= preprocessing.random_crop(self._crop_shape,
                                                                              [left_image, right_image,
                                                                               gt_image,line_image])
        else:
            (left_image, right_image, gt_image, laser_image,line_image) = [
                tf.image.resize_image_with_crop_or_pad(x, self._crop_shape[0], self._crop_shape[1]) for x in
                [left_image, right_image, gt_image, laser_image,line_image]]

        if self._augment:
            left_image, right_image = preprocessing.augment(left_image, right_image)
        return [left_image, right_image, gt_image, laser_image,line_image] if sgbm_image != None else [left_image, right_image,
                                                                                              gt_image,line_image]

    def _build_input_pipeline(self):
        self.pk_files = read_list_file(self._path_file)
        self._couples = [[a] for a in self.pk_files]
        nross = nori.Fetcher()
        r = pickle.loads(nross.get(self.pk_files[0]))

        self._usePfm = 'name' not in r.keys()
        #print(r.keys())
        self._sgbm = 'sgbm' in r.keys()
        # flags
        # self._usePfm = gt_files[0].endswith('pfm') or gt_files[0].endswith('PFM')
        #
        # self._usePfm_las = laser_files[0].endswith('pfm') or laser_files[0].endswith('PFM')
        # if not self._usePfm:
        #     gg = cv2.imread(gt_files[0], -1)
        #     self._double_prec_gt = (gg.dtype == np.uint16)  ####??32位咋办，用在哪里了
        # if not self._usePfm_las:
        #     hh = cv2.imread(laser_files[0], -1)
        #     self._double_prec_las = (hh.dtype == np.uint16)  ####??32位咋办，用在哪里了

        # create dataset

        dataset = tf.data.Dataset.from_tensor_slices(self._couples).repeat(self._num_epochs)
        if self._shuffle:
            dataset = dataset.shuffle(self._batch_size * 50)

        # load images
        # dataset = dataset.flat_map(
        #     lambda filename:
        #     tf.data.TextLineDataset(filename)).map(self._load_image)
        dataset = dataset.map(self._load_image)

        # transform data
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=30)

        # get iterator and batches
        iterator = dataset.make_one_shot_iterator()
        images = iterator.get_next()  ##计算图，每次都会调用。。
        self._left_batch = images[0]
        self._right_batch = images[1]
        self._gt_batch = images[2]
        if (self._sgbm):
            self._laser_batch = images[3]
        else:
            self._laser_batch = []
        self._line_batch = images[-1]

    ################# PUBLIC METHOD #######################

    def __len__(self):
        return len(self._couples)

    def get_max_steps(self):
        return (len(self) * self._num_epochs) // self._batch_size

    def get_batch(self):
        return self._left_batch, self._right_batch, self._gt_batch, self._laser_batch, self._line_batch

    # def get_couples(self):
    #     return self._couples


