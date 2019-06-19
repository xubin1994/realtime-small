import os
import csv

import os.path
import glob
import fnmatch # pattern matching

split = 'train_laser_myt'




data_dirs = []
left = []
right = []
disp_L = []
csvFile2 = open('csvFile2'+split+'.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
writer = csv.writer(csvFile2)
if split == "train_laser_myt":

    names = sorted([fname for fname in os.listdir('/unsullied/sharefs/jiangying/data/mynteye/left/') if
                    fname in os.listdir('/unsullied/sharefs/jiangying/data/mynteye/right/')])
    names = sorted([fname for fname in names if
                    fname in os.listdir('/unsullied/sharefs/jiangying/data/mynteye/laserpng/left/')])
    #print(names)
    left = ["/unsullied/sharefs/jiangying/data/mynteye/left/" + nam  for nam in names]
    right = ["/unsullied/sharefs/jiangying/data/mynteye/right/" + nam for nam in names]
    disp = ["/unsullied/sharefs/jiangying/data/mynteye/laserpng/left/" + nam for nam in names]
    las = disp
    #disp_L = sorted([os.path.join(laser_dir, fname + '.png') for fname in frames])
    writer.writerows(zip(left, right, disp,las))