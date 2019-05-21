import tensorflow as tf
import numpy as np
import cv2
import re
import os
from collections import defaultdict

from Data_utils import preprocessing


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


def read_list_file(path_file):
    """
    Read dataset description file encoded as left;right;disp;conf
    Args:
        path_file: path to the file encoding the database
    Returns:
        [left,right,gt,conf] 4 list containing the images to be loaded
    """
    with open(path_file, 'r') as f_in:
        lines = f_in.readlines()
    lines = [x for x in lines if not x.strip()[0] == '#']
    left_file_list = []
    right_file_list = []
    gt_file_list = []
    conf_file_list = []
    for l in lines:
        to_load = re.split(',|;', l.strip())
        left_file_list.append(to_load[0])
        right_file_list.append(to_load[1])
        if len(to_load) > 2:
            gt_file_list.append(to_load[2])
        if len(to_load) > 3:
            conf_file_list.append(to_load[3])  ######没有就返回空值
            #####
    return left_file_list, right_file_list, gt_file_list, conf_file_list


def read_image_from_disc(image_path, shape=None, dtype=tf.uint8):
    """
    Create a queue to hoold the paths of files to be loaded, then create meta op to read and decode image
    Args:
        image_path: metaop with path of the image to be loaded
        shape: optional shape for the image
    Returns:
        meta_op with image_data
    """
    image_raw = tf.read_file(image_path)
    if dtype == tf.uint8:
        image = tf.image.decode_image(image_raw)
    else:
        image = tf.image.decode_png(image_raw, dtype=dtype)
    if shape is None:
        image.set_shape([None, None, 3])
    else:
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

        if not os.path.exists(path_file):
            raise Exception('File not found during dataset construction')

        self._path_file = path_file
        self._batch_size = batch_size
        self._crop_shape = crop_shape
        self._num_epochs = num_epochs
        self._augment = augment
        self._shuffle = shuffle
        self._is_training = is_training

        self._build_input_pipeline()

    def _load_image(self, files):
        left_file_name = files[0]
        right_file_name = files[1]
        gt_file_name = files[2]
        if (files.shape[0] > 3):
            laser_file_name = files[3]
        left_image = read_image_from_disc(left_file_name)
        right_image = read_image_from_disc(right_file_name)
        if self._usePfm:
            gt_image = tf.py_func(lambda x: readPFM(x)[0], [gt_file_name], tf.float32)
            gt_image.set_shape([None, None, 1])
        else:
            read_type = tf.uint16 if self._double_prec_gt else tf.uint8
            gt_image = read_image_from_disc(gt_file_name, shape=[None, None, 1], dtype=read_type)
            gt_image = tf.cast(gt_image, tf.float32)
            # if read_type == tf.uint16:#####不是即可
            if self._double_prec_gt:
                gt_image = gt_image / 256.0
        if (files.shape[0] > 3):
            read_type = tf.uint16 if self._double_prec_las else tf.uint8
            laser_image = read_image_from_disc(laser_file_name, shape=[None, None, 1], dtype=read_type)
            laser_image = tf.cast(laser_image, tf.float32)
            laser_image = tf.where(tf.greater(laser_image, 254.5), laser_image,
                                   tf.zeros_like(laser_image, dtype=tf.float32))
            if self._double_prec_las:  #####不是即可
                laser_image = laser_image / 256.0

        if self._usePfm:##same as gt
            line_image = tf.py_func(lambda x: readPFM(x)[0], [gt_file_name], tf.float32)
            line_image.set_shape([540, 960, 1])
            print(line_image.get_shape())
            a = line_image.get_shape().as_list()[0]
            b = line_image.get_shape().as_list()[1]
            c = line_image.get_shape().as_list()[2]
            part1 = tf.zeros([270,b,c])
            part2 = tf.zeros([a-272,b,c])
            val = line_image[270:272]
            line_image= tf.concat([part1, val, part2], axis=0)
        else:
            read_type = tf.uint16 if self._double_prec_las else tf.uint8
            line_image = read_image_from_disc(gt_file_name, shape=[None, None, 1], dtype=read_type)
            line_image = tf.cast(line_image, tf.float32)
            part1 = tf.zeros([270, line_image.get_shape().as_list()[1], line_image.get_shape().as_list()[2]])
            part2 = tf.zeros([line_image.get_shape().as_list()[0] - 272, line_image.get_shape().as_list()[1],
                              line_image.get_shape().as_list()[2]])
            val = line_image[270:272]
            line_image = tf.concat([part1, val, part2], axis=0)
            if self._double_prec_las:  #####不是即可
                line_image = line_image / 256.0

        # crop gt to fit with image (SGM add some paddings who know why...)
        gt_image = gt_image[:, :tf.shape(left_image)[1], :]
        line_image = line_image[:, :tf.shape(left_image)[1], :]
        if (files.shape[0] > 3):
            laser_image = laser_image[:, :tf.shape(left_image)[1], :]

        if self._is_training:
            if (files.shape[0] > 3):
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
        return [left_image, right_image, gt_image, laser_image,line_image] if (files.shape[0] > 3) else [left_image, right_image,
                                                                                              gt_image,line_image]

    def _build_input_pipeline(self):
        left_files, right_files, gt_files, laser_files = read_list_file(self._path_file)
        self._couples = [[l, r, gt, las] for l, r, gt, las in
                         zip(left_files, right_files, gt_files, laser_files)] if laser_files != [] else [[l, r, gt] for
                                                                                                         l, r, gt in
                                                                                                         zip(left_files,
                                                                                                             right_files,
                                                                                                             gt_files)]
        # flags
        self._usePfm = gt_files[0].endswith('pfm') or gt_files[0].endswith('PFM')
        self._usePfm_las = laser_files[0].endswith('pfm') or laser_files[0].endswith('PFM')
        if not self._usePfm:
            gg = cv2.imread(gt_files[0], -1)
            self._double_prec_gt = (gg.dtype == np.uint16)  ####??32位咋办，用在哪里了
        if not self._usePfm_las:
            hh = cv2.imread(laser_files[0], -1)
            self._double_prec_las = (hh.dtype == np.uint16)  ####??32位咋办，用在哪里了

        # create dataset
        dataset = tf.data.Dataset.from_tensor_slices(self._couples).repeat(self._num_epochs)
        if self._shuffle:
            dataset = dataset.shuffle(self._batch_size * 50)

        # load images
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
        if (laser_files != []):
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

    def get_couples(self):
        return self._couples

