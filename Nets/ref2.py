import torch.nn as nn
import sys
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import tensorflow as tf
from torch.nn.modules.conv import _ConvNd
import numpy as np
def nconv_max_pool(inputx, inputc, ksize, strides,padding="SAME", name='nconv-maxpool'):
    with tf.variable_scope(name):
        outputc, arg_max = tf.nn.max_pool_with_argmax(input=inputc,ksize=ksize,strides=strides,padding=padding)

        outputc+=1.0
        shape=tf.shape(outputc)
        outputx = tf.zeros(shape)
        outputx=tf.reshape(tf.gather(tf.reshape(inputx,[-1]),arg_max),shape)
        print_op = tf.print(outputx, output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
            outputc-=1.0
        #err=tf.reduce_sum(tf.square(tf.subtract(output,output1)))
        outputc /= 4.0
        print_op = tf.print(outputc, output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
            outputx+=1.0
    return outputx-1.0, outputc

def nconv2d(W,b,x,c, kernel_shape, strides=1,dilations=[1,1,1,1], activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='nconv', reuse=False, wName='weights', bName='bias', batch_norm=False, training=False):##same!!
    with tf.variable_scope(name, reuse=reuse):#kernel:3*3*输入通道数×输出通道数  no dilations
        eps = 1e-20
        padding = 'SAME'
        #W = tf.get_variable(wName, kernel_shape, initializer=INITIALIZER_CONV)
        #b = tf.get_variable(bName, kernel_shape[3], initializer=INITIALIZER_BIAS_NCONV)
        #W = 0.1*tf.nn.softplus(0.1*W)
        W = 0.1 * tf.log(tf.exp(10 * W) + 1.0)
        print_op = tf.print(W, output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
            denom = tf.nn.conv2d(c, W, strides=[1, strides, strides, 1], padding=padding)
            nomin = tf.nn.conv2d(x*c, W, strides=[1, strides, strides, 1], padding=padding)
            nconv = nomin / (denom + eps)

        nconv = tf.nn.bias_add(nconv, b)###要写
        cout = denom

        sz = cout.shape
        cout = tf.reshape(cout,[sz[0], -1, sz[-1]])#Output: :math:`(N, C_{out}, H_{out}, W_{out})`torch, here  tf[b, i, j, k(channelout)]

        k = W
        k_sz = k.shape#(out_channels, in_channels, kernel_size[0], kernel_size[1]) for torch weight
        k = tf.reshape(k,[-1,k_sz[-1]])#here  filter[di, dj, q, k],weight就是kernel

        s = tf.reduce_sum(k, axis=0, keepdims=True)

        cout = cout / (s + eps)
        cout = tf.reshape(cout,sz)  ###归一化？？

        print_op = tf.print(nconv, output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
            print_op2 = tf.print(cout, output_stream=sys.stdout)
        with tf.control_dependencies([print_op2]):
            cout+=1.0
            if batch_norm:
                nconv = tf.layers.batch_normalization(nconv,training=training,momentum=0.99)
                cout = tf.layers.batch_normalization(cout,training=training,momentum=0.99)

    cout-=1.0
    return W,nconv, cout


class NConv2d(_ConvNd):
    def __init__(self, weight,in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, bias=True):

        # Call _ConvNd constructor
        super(NConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, 0,
                                      groups, bias)

        self.eps = 1e-20
        self.pos_fn = pos_fn  ##gamma
        self.init_method = init_method

        # Initialize weights and bias
        self.init_parameters()
        self.weight = torch.nn.Parameter(1/10.0*torch.log(torch.exp(0.1*weight)+1.0))

        '''if self.pos_fn is not None:F.softplus(weight, beta=10)
            EnforcePos.apply(self, 'weight', pos_fn)'''
        print(self.weight)

        '''if self.pos_fn is not None:F.softplus(weight, beta=10)
            EnforcePos.apply(self, 'weight', pos_fn)'''
        self.weight = torch.nn.Parameter(F.softplus(weight, beta=10))
        print(self.weight)

    def forward(self, data, conf):

        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nomin = F.conv2d(data * conf, self.weight, None, self.stride,  ##点乘之后
                         self.padding, self.dilation, self.groups)
        nconv = nomin / (denom + self.eps)  ##conf为分母

        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1, sz, 1, 1)  # 第二维为输出通道
        b = b.expand_as(nconv)
        nconv += b

        # Propagate confidence
        cout = denom

        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)

        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)

        cout = cout / (s+ self.eps)
        cout = cout.view(sz)  ###归一化？？

        return nconv, cout

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k':  # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        '''
        elif self.init_method == 'p':  # Poisson
            mu = self.kernel_size[0] / 2
            dist = poisson(mu)
            x = np.arange(0, self.kernel_size[0])
            y = np.expand_dims(dist.pmf(x), 1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.Tensor(w).type_as(self.weight)
            w = torch.unsqueeze(w, 0)
            w = torch.unsqueeze(w, 1)
            w = w.repeat(self.out_channels, 1, 1, 1)
            w = w.repeat(1, self.in_channels, 1, 1)
            self.weight.data = w + torch.rand(w.shape)'''

        # Init bias
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels) + 0.01)


# Non-negativity enforcement class
class EnforcePos(object):
    def __init__(self, pos_fn, name):
        self.name = name
        self.pos_fn = pos_fn

    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(pos_fn, name)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, inputs):

        weight = getattr(module, self.name)
        weight.data = self._pos(weight).data


    def _pos(self, p):
        pos_fn = self.pos_fn.lower()
        if pos_fn == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0], p_sz[1], -1)
            p = F.softmax(p, -1)
            return p.view(p_sz)
        elif pos_fn == 'exp':
            return torch.exp(p)
        elif pos_fn == 'softplus':
            return F.softplus(p, beta=10)
        elif pos_fn == 'sigmoid':
            return F.sigmoid(p)
        else:
            print('Undefined positive function!')
            return


class CNN(nn.Module):

    def __init__(self, weight, pos_fn=None, num_channels=2):
        super().__init__()

        self.pos_fn = pos_fn

        self.nconv1 = NConv2d(torch.nn.Parameter(weight),1, num_channels, (5, 5), pos_fn, 'p', padding=2)

    def forward(self, x0, c0):
        x1, c1 = self.nconv1(x0, c0)
        ds = 2
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)  ##顺便返回了娶谁的id，求导正好用上的

        x1_ds = torch.zeros(c1_ds.size()).cuda()
        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i, j, :, :] = x1[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])  ##选择可信的点
        c1_ds /= 4  ##归一化

        return x1_ds, c1_ds

if __name__ == '__main__':
    #weight_matrix = torch.randn(1, 1, 228, 304)
    blur_matrix = torch.randn(1, 1, 228, 304)
    conf = torch.randn(1, 1, 228, 304)
    #spn_kernel = 3
    weight = -1*torch.randn(2,1,5, 5)
    #weight[0,0,0,0]=0.0
    #torch.nn.init.kaiming_uniform_(weight)
    cnn = CNN(weight)
    cnn.train()
    d0 = cnn(blur_matrix,conf)

    b = np.zeros(2) + 0.01
    weight = np.transpose(np.array(weight),(2,3,1,0))
    input_layer = np.array(blur_matrix[:, 0, :, :].unsqueeze(-1))
    input_conf = np.array(conf[:, 0, :, :].unsqueeze(-1))
    W,d1,c = nconv2d(weight, b, input_layer, input_conf, [
                    5, 5, 1, 2], name='pre-nconv-1')
    ds = [1, 2, 2, 1]
    d11, c2=nconv_max_pool(d1, c,ksize=ds, strides = ds, name='down2-maxpool')
    print(d0[0].permute(0,2,3,1))
    print(d0[1].permute(0, 2, 3, 1))
    print(d1)
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        sess.run([W,d11,c2])#buyiqishaoda