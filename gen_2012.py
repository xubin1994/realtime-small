import os
import csv

import os.path
import glob
import fnmatch # pattern matching

split = "test_laser_2012"
dates = ['2011_10_03', '2011_09_26','2011_09_29','2011_09_30','2011_09_28']#####,  , ]
val_set = ['2011_09_26_drive_0002_sync', '2011_09_26_drive_0014_sync',
                '2011_09_26_drive_0023_sync','2011_09_26_drive_0079_sync']
test_set = []
'''
test_set = ['2011_10_03_drive_0058_sync', '2011_09_26_drive_0009_sync', '2011_09_30_drive_0072_sync',
            '2011_09_26_drive_0119_sync'
    , '2011_09_28_drive_0225_sync', '2011_09_29_drive_0108_sync']  ##lack velo  2011_09_29_drive_0108_sync'''

pright = '/unsullied/sharefs/jiangying/data/k2012/training/colored_0/'
pleft ='/unsullied/sharefs/jiangying/data/k2012/training/colored_0/'
plim = '/unsullied/sharefs/jiangying/data/k2012/training/disp_0/'

data_dirs = []
left = []
right = []
disp_L = []
csvFile2 = open('csvFile2'+split+'.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
writer = csv.writer(csvFile2)
if split == "train_laser_2012":

    names = sorted([fname[:-7] for fname in os.listdir('/unsullied/sharefs/jiangying/data/k2012/training/colored_0/') if
                    fname[-5] != '1'])
    names = sorted([fname for fname in names if fname+'_11.png' in os.listdir('/unsullied/sharefs/jiangying/data/k2012/training/colored_0/')])
    #print(names)
    left = ["/unsullied/sharefs/jiangying/data/k2012/training/colored_0/" + nam + "_10.png" for nam in names]
    right = ["/unsullied/sharefs/jiangying/data/k2012/training/colored_0/" + nam + "_11.png" for nam in names]
    disp = ["/unsullied/sharefs/jiangying/data/k2012/training/disp_noc/" + nam + "_10.png" for nam in names]
    las = ['/unsullied/sharefs/jiangying/data/k2012/training/disp_0/' + nam + "_10.png" for nam in names]
    #disp_L = sorted([os.path.join(laser_dir, fname + '.png') for fname in frames])
    writer.writerows(zip(left, right, disp,las))
elif split == "test_laser_2012":
        names = sorted(
            [fname[:-7] for fname in os.listdir('/unsullied/sharefs/jiangying/data/k2012/testing/colored_0/') if
             fname[-5] != '1'])
        names = sorted([fname for fname in names if
                        fname + '_11.png' in os.listdir('/unsullied/sharefs/jiangying/data/k2012/testing/colored_0/')])
        # print(names)
        left = ["/unsullied/sharefs/jiangying/data/k2012/testing/colored_0/" + nam + "_10.png" for nam in names]
        right = ["/unsullied/sharefs/jiangying/data/k2012/testing/colored_0/" + nam + "_11.png" for nam in names]
        disp = ["/unsullied/sharefs/jiangying/data/k2012/testing/disp_0/" + nam + "_10.png" for nam in names]
        las = ['/unsullied/sharefs/jiangying/data/k2012/testing/disp_0/' + nam + "_10.png" for nam in names]
        # disp_L = sorted([os.path.join(laser_dir, fname + '.png') for fname in frames])
        writer.writerows(zip(left, right, disp,las))#, disp
elif split == "test_laser_line":
    names = sorted(
        [fname[:-7] for fname in os.listdir('/unsullied/sharefs/jiangying/data/K2015/k2015/training/image_2') if
         fname[-5] != '1'])
    names = sorted([fname for fname in names if fname + '_11.png' in os.listdir(
        '/unsullied/sharefs/jiangying/data/K2015/k2015/training/image_2')])
    # print(names)
    left = ["/unsullied/sharefs/jiangying/data/K2015/k2015/training/image_2/" + nam + "_10.png" for nam in names]
    right = ["/unsullied/sharefs/jiangying/data/K2015/k2015/training/image_2/" + nam + "_11.png" for nam in names]
    disp = ["/unsullied/sharefs/jiangying/data/K2015/k2015/training/disp_noc_0_line/" + nam + "_10.png" for nam in names]
    las = ['/unsullied/sharefs/jiangying/data/K2015/k2015/training/disp_2/' + nam + "_10.png" for nam in names]
    # disp_L = sorted([os.path.join(laser_dir, fname + '.png') for fname in frames])
    writer.writerows(zip(left, right, disp, las))
elif split == 'train':
    #laser0_dir = '/unsullied/sharefs/jiangying/data/Kitti_velo_origin/' + split
    #dirs = os.listdir(laser0_dir)
    for date in dates:
        root_dir = '/unsullied/sharefs/_research_slam/data/KITTI/' + date
        laser0_dir = '/unsullied/sharefs/jiangying/data/KittiDisppng/' + date
        dirs = os.listdir(laser0_dir)
        data_dirs = [a for a in os.listdir(root_dir) if a in dirs and a not in val_set]
        for data_dir in data_dirs:
            left_dir = os.path.join(root_dir,data_dir, 'image_02/data/')
            right_dir = os.path.join(root_dir, data_dir, 'image_03/data/')
            laser_dir = os.path.join(laser0_dir,data_dir, 'laser_02/data/')
            left_frame = [a[:-4] for a in os.listdir(left_dir)]
            las_frame = [a[:-4] for a in os.listdir(laser_dir)]
            #las_frame = [a[:-5] for a in os.listdir(laser_dir)]
            #lasR_frame = [a[:-5] for a in os.listdir(laserR_dir)]
            frames = [a for a in left_frame if (a in las_frame)] #and a in lasR_frame)]
            if len(frames) == 0:
                print(laser_dir)
                print(left_frame)
                print(las_frame)
            left = sorted([os.path.join(left_dir, fname + '.png') for fname in frames])
            right = sorted([os.path.join(right_dir, fname + '.png') for fname in frames])
            disp_L = sorted([os.path.join(laser_dir, fname + '.png') for fname in frames])
            writer.writerows(zip(left,right,disp_L))

elif split == 'train_laser':
    #laser0_dir = '/unsullied/sharefs/jiangying/data/Kitti_velo_origin/' + split
    #dirs = os.listdir(laser0_dir)
    for date in dates:
        root_dir = '/unsullied/sharefs/_research_slam/data/KITTI/' + date
        laser0_dir = '/unsullied/sharefs/jiangying/data/KittiDisppng/' + date
        laser1_dir = '/unsullied/sharefs/jiangying/data/KittiSGBM/' + date
        dirs = os.listdir(laser0_dir)
        data_dirs = [a for a in os.listdir(root_dir) if a in dirs and a not in val_set]
        for data_dir in data_dirs:
            left_dir = os.path.join(root_dir,data_dir, 'image_02/data/')
            right_dir = os.path.join(root_dir, data_dir, 'image_03/data/')
            laser_dir = os.path.join(laser0_dir,data_dir, 'laser_02/data/')
            laser11_dir = os.path.join(laser1_dir, data_dir, 'disp_02/')
            left_frame = [a[:-4] for a in os.listdir(left_dir)]
            las_frame = [a[:-4] for a in os.listdir(laser_dir)]
            las11_frame = [a[:-4] for a in os.listdir(laser11_dir) if os.path.isfile(os.path.join(laser11_dir, a))]
            #las_frame = [a[:-5] for a in os.listdir(laser_dir)]
            #lasR_frame = [a[:-5] for a in os.listdir(laserR_dir)]
            frames = [a for a in left_frame if (a in las_frame) and 'data'+a in las11_frame]
            if len(frames) == 0:
                print(laser_dir)
                print(left_frame)
                print(las_frame)
            left = sorted([os.path.join(left_dir, fname + '.png') for fname in frames])
            right = sorted([os.path.join(right_dir, fname + '.png') for fname in frames])
            disp_L = sorted([os.path.join(laser_dir, fname + '.png') for fname in frames])
            las_L = sorted([os.path.join(laser11_dir, 'data'+fname + '.png') for fname in frames])
            writer.writerows(zip(left,right,disp_L,las_L))

elif split == 'train_laser_line':
    #laser0_dir = '/unsullied/sharefs/jiangying/data/Kitti_velo_origin/' + split
    #dirs = os.listdir(laser0_dir)
    for date in dates:
        root_dir = '/unsullied/sharefs/_research_slam/data/KITTI/' + date
        laser0_dir = '/unsullied/sharefs/jiangying/data/KittiDispLinepng/' + date
        laser1_dir = '/unsullied/sharefs/jiangying/data/KittiSGBM/' + date
        dirs = os.listdir(laser0_dir)
        data_dirs = [a for a in os.listdir(root_dir) if a in dirs and a not in val_set]
        for data_dir in data_dirs:
            left_dir = os.path.join(root_dir,data_dir, 'image_02/data/')
            right_dir = os.path.join(root_dir, data_dir, 'image_03/data/')
            laser_dir = os.path.join(laser0_dir,data_dir, 'laser_02/data/')
            laser11_dir = os.path.join(laser1_dir, data_dir, 'disp_02/')
            left_frame = [a[:-4] for a in os.listdir(left_dir)]
            las_frame = [a[:-4] for a in os.listdir(laser_dir)]
            las11_frame = [a[:-4] for a in os.listdir(laser11_dir) if os.path.isfile(os.path.join(laser11_dir, a))]
            #las_frame = [a[:-5] for a in os.listdir(laser_dir)]
            #lasR_frame = [a[:-5] for a in os.listdir(laserR_dir)]
            frames = [a for a in left_frame if (a in las_frame) and 'data'+a in las11_frame]
            if len(frames) == 0:
                print(laser_dir)
                print(left_frame)
                print(las_frame)
            left = sorted([os.path.join(left_dir, fname + '.png') for fname in frames])
            right = sorted([os.path.join(right_dir, fname + '.png') for fname in frames])
            disp_L = sorted([os.path.join(laser_dir, fname + '.png') for fname in frames])
            las_L = sorted([os.path.join(laser11_dir, 'data'+fname + '.png') for fname in frames])
            writer.writerows(zip(left,right,disp_L,las_L))

csvFile2.close()