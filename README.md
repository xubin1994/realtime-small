# realtimelinecspn, disp+cspn
双目视差估计与单线激光数据的融合(9月以后的工作）

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
