import cv2
import math
import numpy as np
import os
n = 3772
pics = []
with open('/home/jiangying/data.txt','r') as f:
    for line in f:
        line = line.lstrip()
        if (line.startswith('header')):
            pic = {'secs':0, 'nsecs':0, 'pts4d':[]}
        elif(line.startswith('secs')):
            pic['secs']=int(line[line.find(':')+2:])
        elif (line.startswith('nsecs')):
            pic['nsecs'] = int(line[line.find(':') + 2:])
        elif (line.startswith('range_min')):
            rmin = float(line[line.find(':') + 2:])
        elif (line.startswith('range_max')):
            rmax = float(line[line.find(':') + 2:])
        elif (line.startswith('angle_min')):
            amin = float(line[line.find(':') + 2:])
        elif (line.startswith('angle_increment')):
            inc = float(line[line.find(':') + 2:])
        elif (line.startswith('ranges')):
            tup = eval(line[line.find(':') + 2:])
            for i, r in enumerate(tup):
                if(r>rmax):
                    continue
                if(r<rmin):
                    continue
                a = amin+i*inc;
                x = r*math.cos(a) #shendu
                y = r*math.sin(a) #left
                z = 0
                d = 1
                pic['pts4d'].append([x,y,z,d])
        elif (line.startswith('intensities')):
            pics.append(pic)
            del pic
print(len(pics))
finalpic = []
p = '/unsullied/sharefs/jiangying/data/mynteye/velo'
with open('/unsullied/sharefs/_research_slam/data/mynteye/mynteye/times.txt','r') as f:
    cnt = 0
    for i, line in enumerate(f):
        t = float(line[3:])
        tp = float(str(pics[cnt]['secs'])[3:]+'.'+str(pics[cnt]['nsecs']))
        tpp = float(str(pics[cnt+1]['secs'])[3:]+'.'+str(pics[cnt+1]['nsecs']))
        while (t>tpp and cnt < n):
            tp = tpp;
            cnt += 1
            tpp = float(str(pics[cnt + 1]['secs'])[3:] + '.' + str(pics[cnt + 1]['nsecs']))
        # t <= tpp
        if(cnt == n):
            np.save(os.path.join(p,'{:06d}.npy'.format(i)), np.array(pics[n-1]['pts4d']))
            continue
        elif (t<tp):## t small, choose tp
            np.save(os.path.join(p, '{:06d}.npy'.format(i)), np.array(pics[cnt]['pts4d']))
        elif (tp <= t <= tpp and tpp - t > t - tp):  ## choose tp
            np.save(os.path.join(p, '{:06d}.npy'.format(i)), np.array(pics[cnt]['pts4d']))
        elif (tp <= t <= tpp and tpp - t <= t - tp):  ## choose tp
            np.save(os.path.join(p, '{:06d}.npy'.format(i)), np.array(pics[cnt+1]['pts4d']))







