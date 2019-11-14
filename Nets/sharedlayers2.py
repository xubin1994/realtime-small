import tensorflow as tf
import os

INITIALIZER_CONV = tf.contrib.layers.xavier_initializer()
INITIALIZER_BIAS = tf.constant_initializer(0.0)
INITIALIZER_BIAS_NCONV = tf.constant_initializer(0.01)
MODE = 'TF'
LEAKY_RELU = lambda x: tf.maximum(0.2 * x, x)
RELU= lambda x: tf.maximum(0.0, x)
NO_ACTIVATION = lambda x: x
####################################################
#### Uncomment lines below for cuda correlation ####
####################################################

#REPO_DIR = os.path.dirname(os.path.abspath(__file__))
#shift_corr_module = tf.load_op_library(os.path.join(REPO_DIR, 'Native/shift_corr.so'))

#@tf.RegisterGradient("ShiftCorr")
#def _ShiftCorrOpGrad(op, grad):
#	return shift_corr_module.shift_corr_grad(op.inputs[0], op.inputs[1], grad, max_disp=op.get_attr('max_disp'))

# MODE='CUDA'

#######################################################

def correlation(x,y,max_disp, name='corr', mode=MODE,stride=1):
	if mode == 'TF':
		return correlation_tf(x,y,max_disp,name=name,stride=stride)
	else:
		if stride!=1:
			raise Exception('Cuda implementation cannot handle stride different than 1')
		return correlation_native(x,y,max_disp,name=name)

def correlation_native(x, y, max_disp, name='corr'):
	with tf.variable_scope(name):
		input_shape = x.get_shape().as_list()
		x = tf.pad(x, [[0, 0], [0, 0], [max_disp, max_disp], [0, 0]], "CONSTANT")
		y = tf.pad(y, [[0, 0], [0, 0], [max_disp, max_disp], [0, 0]], "CONSTANT")
		corr = shift_corr_module.shift_corr(x, y, max_disp=max_disp)
		corr = tf.transpose(corr, perm=[0, 2, 3, 1])
		corr.set_shape([input_shape[0],input_shape[1],input_shape[2],2*max_disp+1])
		return corr

def correlation_tf(x, y, max_disp, stride=1, name='corr'):
	with tf.variable_scope(name):
		corr_tensors = []
		y_shape = tf.shape(y)
		y_feature = tf.pad(y,[[0,0],[0,0],[max_disp,max_disp],[0,0]])#补全边界
		for i in range(-max_disp, max_disp+1,stride):
			shifted = tf.slice(y_feature, [0, 0, i + max_disp, 0], [-1, y_shape[1], y_shape[2], -1])#每一种偏移都尝试切下来
			corr_tensors.append(tf.reduce_mean(shifted*x, axis=-1, keepdims=True))#3个颜色求均值？最后一位channel?

		result = tf.concat(corr_tensors,axis=-1)
		return result

def resnext_block(name, data, stride, channels, group, has_proj=False):
    proj = data
    bottleneck = channels // 4
    assert (bottleneck % group == 0) and (bottleneck / group) % 4 == 0, (bottleneck, group)
    if has_proj:
        if stride == 2:
            proj = tf.layers.average_pooling2d(proj,pool_size=2,strides=2,name="%s-dsp"%(name))###padding = valid
        proj = conv2d(proj, name="%s-shortcut"%(name), kernel_shape=[1, 1, proj.get_shape().as_list()[3],
                                                                   channels], strides=1,
                     batch_norm=True, activation=NO_ACTIVATION)
    x = conv2d(data, name="%s-1x1_shrink" % (name), kernel_shape=[1, 1, data.get_shape().as_list()[3],
                                                               bottleneck], strides=1, batch_norm=True,
                activation=LEAKY_RELU)


    x = grouped_conv2d(x, name="%s-3x3" % (name),num_groups=group, kernel_shape=[3, 3, x.get_shape().as_list()[3],
                                                                  bottleneck], strides=stride, batch_norm=True,
               activation=LEAKY_RELU)
    #x = swishnorm('%s-swish1'%(name), x)
    x = conv2d(x, name="%s-1x1_expand" % (name), kernel_shape=[1, 1, x.get_shape().as_list()[3],
                                                                   channels], strides=1,
                  batch_norm=True, activation=NO_ACTIVATION)

    assert x.get_shape() == proj.get_shape(), (x.get_shape(), proj.get_shape())
    x = x + proj
    x = RELU(x)
    return x


def conv_block(name, data, ks, output_channel):
    x0 = conv2d(data, name = "%s-conv00" % (name), kernel_shape = [1, ks, data.get_shape().as_list()[3],
                                                         output_channel], strides = 1,  batch_norm = True, activation = LEAKY_RELU)
    x0 = conv2d(x0, name="%s-conv01" % (name), kernel_shape=[ ks,1, x0.get_shape().as_list()[3],
                                                               output_channel], strides=1,
                 batch_norm=True, activation=NO_ACTIVATION)

    x1 = conv2d(data, name="%s-conv10" % (name), kernel_shape=[ks,1,  data.get_shape().as_list()[3],
                                                               output_channel], strides=1,
                batch_norm=True, activation=LEAKY_RELU)
    x1 = conv2d(x1, name="%s-conv11" % (name), kernel_shape=[1, ks, x1.get_shape().as_list()[3],
                                                             output_channel], strides=1,
                batch_norm=True, activation=NO_ACTIVATION)

    x = x0+x1
    return x


def refine_block(name, data, ks, output_channel):
    x = conv2d(data,name = "%s-refine0"%(name),  kernel_shape=[ks, ks, data.get_shape().as_list()[3], output_channel], strides=1,   batch_norm=True, activation=LEAKY_RELU)
    x = conv2d(x,name = "%s-refine1"%(name),  kernel_shape=[ks, ks, x.get_shape().as_list()[3], output_channel], strides=1,  batch_norm=True, activation=NO_ACTIVATION)
    x = x + data
    return x
def make_encoder(data):
    # 640
    print(data.get_shape())
    #data.get_shape().as_list()[3]
    conv0 = conv2d(data, name="conv00", kernel_shape=[3, 3, 9, 16],
           strides=2, batch_norm=True, activation=LEAKY_RELU)
    # conv0 = swishnorm('conv00-swish', conv0)

    base_ch = 32
    # 320
    conv01 = resnext_block("g03_0", conv0, channels=base_ch * 2, group=2, stride=2, has_proj=True)
    conv01 = resnext_block("g03_1", conv01, channels=base_ch * 2, group=2, stride=1, has_proj=False)
    conv01 = resnext_block("g03_2", conv01, channels=base_ch * 2, group=2, stride=1, has_proj=False)
    conv01 = resnext_block("g03_3", conv01, channels=base_ch * 2, group=2, stride=1, has_proj=False)
    pool01 = tf.layers.average_pooling2d(conv0, pool_size=16, strides=16, name='p03')

    # 160
    conv1 = resnext_block("g3_0", conv01, channels=base_ch * 4, group=2, stride=2, has_proj=True)
    conv1 = resnext_block("g3_1", conv1, channels=base_ch * 4, group=2, stride=1, has_proj=False)
    conv1 = resnext_block("g3_2", conv1, channels=base_ch * 4, group=2, stride=1, has_proj=False)
    conv1 = resnext_block("g3_3", conv1, channels=base_ch * 4, group=2, stride=1, has_proj=False)
    # conv1 = resnext_block("g3_4", conv1, channels=base_ch * 4, group=2, stride=1, has_proj=False)
    pool1 = tf.layers.average_pooling2d(conv01, pool_size=8, strides=8, name='p3')

    # 80
    conv2 = resnext_block("g4_0", conv1, channels=base_ch * 8, group=4, stride=2, has_proj=True)
    conv2 = resnext_block("g4_1", conv2, channels=base_ch * 8, group=4, stride=1, has_proj=False)
    conv2 = resnext_block("g4_2", conv2, channels=base_ch * 8, group=4, stride=1, has_proj=False)
    conv2 = resnext_block("g4_3", conv2, channels=base_ch * 8, group=4, stride=1, has_proj=False)
    conv2 = resnext_block("g4_4", conv2, channels=base_ch * 8, group=4, stride=1, has_proj=False)
    conv2 = resnext_block("g4_5", conv2, channels=base_ch * 8, group=4, stride=1, has_proj=False)
    conv2 = resnext_block("g4_6", conv2, channels=base_ch * 8, group=4, stride=1, has_proj=False)
    # conv2 = resnext_block("g4_7", conv2, channels=base_ch*8, group=4, stride=1, has_proj=False)
    pool2 = tf.layers.average_pooling2d(conv1, pool_size=4, strides=4, name='p4')

    # 40
    conv3 = resnext_block("g5_0", conv2, channels=base_ch * 16, group=8, stride=2, has_proj=True)
    conv3 = resnext_block("g5_1", conv3, channels=base_ch * 16, group=8, stride=1, has_proj=False)
    conv3 = resnext_block("g5_2", conv3, channels=base_ch * 16, group=8, stride=1, has_proj=False)
    conv3 = resnext_block("g5_3", conv3, channels=base_ch * 16, group=8, stride=1, has_proj=False)
    conv3 = resnext_block("g5_4", conv3, channels=base_ch * 16, group=8, stride=1, has_proj=False)
    conv3 = resnext_block("g5_5", conv3, channels=base_ch * 16, group=8, stride=1, has_proj=False)
    conv3 = resnext_block("g5_6", conv3, channels=base_ch * 16, group=8, stride=1, has_proj=False)
    conv3 = resnext_block("g5_7", conv3, channels=base_ch * 16, group=8, stride=1, has_proj=False)
    pool3 = tf.layers.average_pooling2d(conv2, pool_size=2, strides=2, name='p5')
    conv3 = tf.concat([pool01, pool1, pool2, pool3, conv3], axis=-1)
    #print(conv3.get_shape())

    return conv3, conv2, conv1, conv01, conv0


def make_network(data):
    conv3, conv2, conv1, conv01, conv0 = make_encoder(data)

    # Dec
    blocks = [conv3, conv2, conv1, conv01, conv0]

    print("*"*30)
    for b in blocks:
        print(b.get_shape())

    up_former = None
    in_ch = [16, 16, 16, 8, 4]
    out_ch = [16, 16, 8, 4, 4]
    is_conv_block = [True, True, False, False, False]
    for i in range(len(blocks)):
        if is_conv_block[i]:
            up_latter = conv_block('score%d' % i, blocks[i], 3, in_ch[i])
        else:
            up_latter = conv2d(blocks[i], name='score%d' % i, kernel_shape=[3, 3, blocks[i].get_shape().as_list()[3],
                                                            in_ch[i]], strides=1,
               batch_norm=True, activation=NO_ACTIVATION)
        if i > 0:
            up_latter = up_latter + up_former
        up_former = refine_block('refine%d' % i, up_latter, 3, in_ch[i])
        if i < len(blocks) - 1:
            up_former = deconv('resize%d2' % i, up_former, kernel_shape=2, output_nr_channel=out_ch[i], padding=0, stride=2, has_bn=True, has_relu=False)

    up0 = deconv('avg', up_former, kernel_shape=2, output_nr_channel=1, padding=0, stride=2, has_bn=False, has_relu=False)
    pred = tf.math.sigmoid(up0) * 192

    return pred

def deconv(name, data, kernel_shape, output_nr_channel, padding, stride=1, has_bn=True, has_relu=False):
    # deconv('avg', up_former, kernel_shape=2, output_nr_channel=1, padding=0, stride=2, has_bn=False, has_relu=False)
    if has_relu:
        activation = RELU
    else:
        activation = NO_ACTIVATION
    kern = [kernel_shape,kernel_shape,output_nr_channel,data.get_shape().as_list()[3] ]
    x = conv2d_transpose(data, kern, stride, activation, name=name,  batch_norm=has_bn)#padding same....

    # def conv2d_transpose(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), name='conv2d',
    #                      reuse=False, wName='weights', bName='bias', batch_norm=False, training=False):
    #     with tf.variable_scope(name, reuse=reuse):
    #         W = tf.get_variable(wName, kernel_shape, initializer=INITIALIZER_CONV)
    #         tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    #         b = tf.get_variable(bName, kernel_shape[2], initializer=INITIALIZER_BIAS)
    #         x_shape = tf.shape(x)
    #         output_shape = [x_shape[0], x_shape[1] * strides, x_shape[2] * strides, kernel_shape[2]]
    #         x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1], padding='SAME')
    #         x = tf.nn.bias_add(x, b)
    #         if batch_norm:
    #             x = tf.layers.batch_normalization(x, training=training, momentum=0.99)
    #         x = activation(x)
    #         return x



    return x


def conv2d(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv2d', reuse=False, wName='weights', bName='bias', batch_norm=True, training=False, bias = True):
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(wName, kernel_shape, initializer=INITIALIZER_CONV)
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        if bias == True:
            b = tf.get_variable(bName, kernel_shape[3], initializer=INITIALIZER_BIAS)

            x = tf.nn.bias_add(x, b)###要写
        if batch_norm:
            x = tf.layers.batch_normalization(x,training=training,momentum=0.99)
        x = activation(x)
        return x

def nconv2d(x,c, kernel_shape, strides=1,dilations=[1,1,1,1], activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='nconv', reuse=False, wName='weights', bName='bias', batch_norm=False, training=False):##same!!
    with tf.variable_scope(name, reuse=reuse):#kernel:3*3*输入通道数×输出通道数  no dilations
        eps = 1e-20
        padding = 'SAME'
        W = tf.get_variable(wName, kernel_shape, initializer=INITIALIZER_CONV)
        b = tf.get_variable(bName, kernel_shape[3], initializer=INITIALIZER_BIAS_NCONV)
        W = 0.1 * tf.log(tf.exp(10 * W) + 1.0)
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

        if batch_norm:
            nconv = tf.layers.batch_normalization(nconv,training=training,momentum=0.99)
            cout = tf.layers.batch_normalization(cout,training=training,momentum=0.99)

        return nconv, cout


def dilated_conv2d(x, kernel_shape, rate=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='dilated_conv', reuse=False, wName='weights', bName='biases',  batch_norm=False, training=False):
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable(
            wName, kernel_shape, initializer=INITIALIZER_CONV)
        biases = tf.get_variable(
            bName, kernel_shape[3], initializer=INITIALIZER_BIAS)
        x = tf.nn.atrous_conv2d(x, weights, rate=rate, padding=padding)
        x = tf.nn.bias_add(x, biases)
        if batch_norm:
            x = tf.layers.batch_normalization(x,training=training,momentum=0.99)
        x = activation(x)
        return x


def conv2d_transpose(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), name='conv2d', reuse=False, wName='weights', bName='bias',  batch_norm=False, training=False):
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(wName, kernel_shape,initializer=INITIALIZER_CONV)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
        b = tf.get_variable(bName, kernel_shape[2], initializer=INITIALIZER_BIAS)
        x_shape = tf.shape(x)
        output_shape = [x_shape[0], x_shape[1] * strides,x_shape[2] * strides, kernel_shape[2]]
        x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        if batch_norm:
            x = tf.layers.batch_normalization(x,training=training,momentum=0.99)
        x = activation(x)
        return x

def depthwise_conv(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv2d', reuse=False, wName='weights', bName='bias', batch_norm=False, training=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable(wName, kernel_shape, initializer=INITIALIZER_CONV)
        b = tf.get_variable(bName, kernel_shape[3]*kernel_shape[2], initializer=INITIALIZER_BIAS)
        x = tf.nn.depthwise_conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        if batch_norm:
            x = tf.layers.batch_normalization(x, training=training,momentum=0.99)
        x = activation(x)
        return x

def separable_conv2d(x, kernel_shape, channel_multiplier=1, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv2d', reuse=False, wName='weights', bName='bias', batch_norm=True, training=False):
    with tf.variable_scope(name, reuse=reuse):
        #detpthwise conv2d
        depthwise_conv_kernel = [kernel_shape[0],kernel_shape[1],kernel_shape[2],channel_multiplier]
        x = depthwise_conv(x,depthwise_conv_kernel,strides=strides,activation=lambda x: tf.maximum(0.1 * x, x),padding=padding,name='depthwise_conv',reuse=reuse,wName=wName,bName=bName,batch_norm=batch_norm, training=training)

        #pointwise_conv
        pointwise_conv_kernel = [1,1,x.get_shape()[-1].value,kernel_shape[-1]]
        x = conv2d(x,pointwise_conv_kernel,strides=strides,activation=activation,padding=padding,name='pointwise_conv',reuse=reuse,wName=wName,bName=bName,batch_norm=batch_norm, training=training)

        return x

def grouped_conv2d(x, kernel_shape, num_groups=1, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv2d', reuse=False, wName='weights', bName='bias', batch_norm=True, training=False):
    with tf.variable_scope(name,reuse=reuse):
        w = tf.get_variable(wName,shape=kernel_shape,initializer=INITIALIZER_CONV)
        #b = tf.get_variable(bName, kernel_shape[3], initializer=INITIALIZER_BIAS)

        input_groups = tf.split(x,num_or_size_splits=num_groups,axis=-1)
        kernel_groups = tf.split(w, num_or_size_splits=num_groups, axis=2)
        #bias_group = tf.split(b,num_or_size_splits=num_groups,axis=-1) 此处木有
        #output_groups = [tf.nn.conv2d(i, k,[1,strides,strides,1],padding=padding)+bb for i, k,bb in zip(input_groups, kernel_groups,bias_group)]
        output_groups = [tf.nn.conv2d(i, k, [1, strides, strides, 1], padding=padding) for i, k in
                         zip(input_groups, kernel_groups)]
		# Concatenate the groups
        x = tf.concat(output_groups,axis=3)##还要concat在最后一个维度
        if batch_norm:
            x = tf.layers.batch_normalization(x,training=training,momentum=0.99)
        x = activation(x)
        return x

def channel_shuffle_inside_group(x, num_groups, name='shuffle'):
    with tf.variable_scope(name):
        _, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
    return output

def nconv_max_pool(inputx, inputc, ksize, strides,padding="SAME", name='nconv-maxpool'):
    with tf.variable_scope(name):
        outputc, arg_max = tf.nn.max_pool_with_argmax(input=inputc,ksize=ksize,strides=strides,padding=padding)
        shape=tf.shape(outputc)
        outputx=tf.reshape(tf.gather(tf.reshape(inputx,[-1]),arg_max),shape)

        #err=tf.reduce_sum(tf.square(tf.subtract(output,output1)))
    return outputx, outputc/4.0
