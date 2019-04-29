import numpy as np
import os
import cv2
import pandas as pd
from collections import Counter
import pickle

import argparse

from collections import Counter
import pickle


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


###############################################################################
#######################  KITTI

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351


def load_gt_disp_kitti(path):
    gt_disparities = []
    n = 0
    names = sorted([fname[:-7] for fname in os.listdir('/unsullied/sharefs/jiangying/data/K2015/k2015/training/image_2/') if fname[-5] != '1'])
    for nam in names:
        disp = cv2.imread("/unsullied/sharefs/jiangying/data/K2015/k2015/training/disp_noc_0/" + nam + "_10.png", -1)
        #print()
        disp = disp.astype(np.float32) / 256
        gt_disparities.append(disp)
        n+=1;
    return n, gt_disparities


def convert_disps_to_depths_kitti(gt_disparities, pred_disparities):
    gt_depths = []
    pred_depths = []
    pred_disparities_resized = []
    print(len(gt_disparities))
    print(len(pred_disparities))
    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_disp = pred_disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(pred_disp)

        mask = gt_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
    return gt_depths, pred_depths, pred_disparities_resized

'''
parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--split', type=str, help='data split, kitti or eigen', required=True)
parser.add_argument('--predicted_disp_path', type=str, help='path to estimated disparities', required=True)
parser.add_argument('--gt_path', type=str, help='path to ground truth disparities', required=True)
parser.add_argument('--min_depth', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth', type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')

args = parser.parse_args()
'''
if __name__ == '__main__':

    pred_disparities = np.load('disparities.npy')

    #if args.split == 'kitti':

    num_samples, gt_disparities = load_gt_disp_kitti('')
    gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities,
                                                                                     pred_disparities)


    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        min_depth=1e-3
        max_depth=80
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth


        gt_disp = gt_disparities[i]
        mask = gt_disp > 0
        pred_disp = pred_disparities_resized[i]

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask],
                                                                                        pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms',
                                                                                  'd1_all', 'a1', 'a2', 'a3'))
print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(),
                                                                                              sq_rel.mean(), rms.mean(),
                                                                                              log_rms.mean(),
                                                                                              d1_all.mean(), a1.mean(),
                                                                                              a2.mean(), a3.mean()))
