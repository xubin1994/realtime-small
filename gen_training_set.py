import os
import csv

import os.path
import glob
import fnmatch # pattern matching

split = 'test'
dates = ['2011_10_03', '2011_09_26','2011_09_29','2011_09_30','2011_09_28']#####,  , ]
val_set = ['2011_09_26_drive_0002_sync', '2011_09_26_drive_0014_sync',
                '2011_09_26_drive_0023_sync','2011_09_26_drive_0079_sync']
test_set = ['2011_10_03_drive_0058_sync', '2011_09_26_drive_0009_sync', '2011_09_30_drive_0072_sync',
            '2011_09_26_drive_0119_sync'
    , '2011_09_28_drive_0225_sync', '2011_09_29_drive_0108_sync']  ##lack velo  2011_09_29_drive_0108_sync



data_dirs = []
left = []
right = []
disp_L = []
csvFile2 = open('csvFile2'+split+'.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
writer = csv.writer(csvFile2)
if split == "test":
    root_dir = '/unsullied/sharefs/jiangying/data/K2015/training/mytrain/'
    laser0_dir = '/unsullied/sharefs/jiangying/data/K2015/training/mytrain/'
    left_dir = os.path.join(root_dir, 'image_02/data/')
    right_dir = os.path.join(root_dir,'image_03/data/')
    gt_dir = os.path.join(root_dir,'image_03/data/')
    #laser_dir = os.path.join(laser0_dir, 'laser_02/data/')
    #left_frame = [a[:-4] for a in os.listdir(left_dir)]
    #las_frame = [a[:-4] for a in os.listdir(laser_dir)]
    # lasR_frame = [a[:-5] for a in os.listdir(laserR_dir)]
    #frames = [a for a in left_frame if (a in las_frame)]  # and a in lasR_frame)]
    #if len(frames) == 0:
    #   print(laser_dir)

    names = sorted([fname[:-7] for fname in os.listdir('/unsullied/sharefs/jiangying/data/training/training/image_2') if
                    fname[-5] != '1'])
    names = sorted([fname for fname in names if fname+'_11.png' in os.listdir('/unsullied/sharefs/jiangying/data/training/training/image_2')])
    #print(names)
    left = ["/unsullied/sharefs/jiangying/data/training/training/image_2/" + nam + "_10.png" for nam in names]
    right = ["/unsullied/sharefs/jiangying/data/training/training/image_2/" + nam + "_11.png" for nam in names]
    disp = ["/unsullied/sharefs/jiangying/data/training/training/disp_noc_0/" + nam + "_10.png" for nam in names]
    #disp_L = sorted([os.path.join(laser_dir, fname + '.png') for fname in frames])
    writer.writerows(zip(left, right, disp))
else:
    laser0_dir = '/unsullied/sharefs/jiangying/data/Kitti_velo_origin/' + split
    dirs = os.listdir(laser0_dir)
    for date in dates:
        root_dir = '/unsullied/sharefs/_research_slam/data/KITTI/' + date
        data_dirs= [a for a in os.listdir(root_dir) if a in dirs]
        laser0_dir = '/unsullied/sharefs/jiangying/data/KittiLaserReal/' + date
        for data_dir in data_dirs:
            left_dir = os.path.join(root_dir,data_dir, 'image_02/data/')
            right_dir = os.path.join(root_dir, data_dir, 'image_03/data/')
            laser_dir = os.path.join(laser0_dir,data_dir, 'proj_depth/velodyne_raw/image_02/')
            left_frame = [a[:-4] for a in os.listdir(left_dir)]
            las_frame = [a[:-4] for a in os.listdir(laser_dir)]
            #lasR_frame = [a[:-5] for a in os.listdir(laserR_dir)]
            frames = [a for a in left_frame if (a in las_frame)] #and a in lasR_frame)]
            if len(frames) == 0:
                print(laser_dir)
            left = sorted([os.path.join(left_dir, fname + '.png') for fname in frames])
            right = sorted([os.path.join(right_dir, fname + '.png') for fname in frames])
            disp_L = sorted([os.path.join(laser_dir, fname + '.png') for fname in frames])
            writer.writerows(zip(left,right,disp_L))

csvFile2.close()