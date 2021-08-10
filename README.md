# realtimelinecspn, disp+cspn
双目视差估计与单线激光数据的融合

#### 一、网络简介
主干部分参考Real-time self-adaptive deep stereo论文。

结构类似UNet，一共6大层，左边Encoder部分为12次卷积，后每两层连出一支做成如上的cost volume，每层卷积6次。最后会经过context net进行refinement。优势在于网络大小很小，目前没有去掉梯度占8.3G显存。

后面添加激光转成的图片，参考Depth Estimation via Affinity Learned with Convolutional Spatial Propagation Network。调整context net的连接学习出8个权重作为扩散系数，最后将扩散到真值的像素替换网络前半部分的视差作为最后的预测。

网络已经在flyingthings3d和KITTI上训练，将迁移到机器人数据上。

代码文件夹中Data_utils/data_reader_old.py需要读取从bag解压的静态图片，用相机的时间戳和激光数据近似对齐后做成激光图片。图片文件由一个csv文件输入，格式为：左图、右图、ground truth和激光。输入图片大小为960*320。

Data_utils/data_reader_norinosgbm.py读取的是我生成的nori数据集。每个数据的格式是左右目、gt和对应的SGBM图片组成的pickle，读取方式从代码里可见。其中左右目存的是cv2.uint8格式，注意后两者是16位没有对应的encode，所以存的是numpy。数据集的nori list是：nori2_flying_test.list, nori2_flying_train.list, nori2_KITTI_test.list和nori2_KITTI_train.list。

#### 二、代码
（一）预处理

rosbag解压后使用若干脚本得到图片（激光的文本转换为点后转换为图片，可能需要去畸变）
需要修改velotopic.py和undistort.py的标定参数，以及laser2pts.py的路径名。
输入图片的路径用gen_xxx.py帮助生成csv。在使用oss时我用make_nori.py生成了kitti和flying_things的数据集，每个数据的格式是左右目、gt和对应的
SGBM图片组成的pickle。数据集在s3://bucketjy/下面的若干文件夹。代码会自动读取。

（二）训练

运行：python3 Train.py

                [-h] [--trainingSet TRAININGSET]
                [--validationSet VALIDATIONSET] [-o PRETRAINED_DIRECTORY]
                [--weights WEIGHTS] [--modelName {Dispnet,MADNet,MADNet_old}]
                [--lr LR] [--imageShape IMAGESHAPE [IMAGESHAPE ...]]
                [--batchSize BATCHSIZE] [--numEpochs NUMEPOCHS] [--augment]
                [--lossWeights LOSSWEIGHTS [LOSSWEIGHTS ...]]
                [--lossType {mean_huber,sum_huber,mean_SSIM_l1,mean_l2,mean_SSIM,ZNCC,cos_similarity,smoothness,sum_l2,mean_l1,sum_l1}]
                [--decayStep DECAYSTEP]

rescspnkitti文件夹为迁移到kitti数据集后的权重

rescspn2 为训练集Flying Things 3D数据集上训好的模型

测试时：python3 Test.py
图片使用原图大小。只用修改输入.csv路径。
测试batchsize=1并保存结果：

（三）效果

在kitti验证集上与Madnet论文相比，EPE 0.89->0.56，D1-all 2.67->1.0以下。

