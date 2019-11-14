import tensorflow as tf
import numpy as np

from Nets import Stereo_net
from Nets import sharedLayers
from Data_utils import preprocessing


class MadNet(Stereo_net.StereoNet):
    _valid_args = [
                      ("left_img", "meta op for left image batch"),
                      ("right_img", "meta op for right image batch"),
                      ("warping", "flag to enable warping"),
                      ("context_net", "flag to enable context_net"),
                      ("radius_d", "size f the patch using for correlation"),
                      ("stride", "stride used for correlation"),
                      ("bulkhead", "flag to stop gradient propagation among different resolution")
                  ] + Stereo_net.StereoNet._valid_args
    _netName = "MADNet"

    def __init__(self, **kwargs):
        """
        Creation of a MadNet for stereo prediction
        """
        super(MadNet, self).__init__(**kwargs)

    def _validate_args(self, args):
        """
        Check that args contains everything that is needed
        Valid Keys for args:
            left_img: left image op
            right_img: right image op
            warping: boolean to enable or disable warping
            context_net: boolean to enable or disable context_net
            radius_d: kernel side used for computing correlation map
            stride: stride used to compute correlation map
        """
        super(MadNet, self)._validate_args(args)
        if ('left_img' not in args) or ('right_img' not in args):
            raise Exception('Missing input op for left and right images')
        if 'warping' not in args:
            print('WARNING: warping flag not setted, setting default True value')
            args['warping'] = True
        if 'context_net' not in args:
            print('WARNING: context_net flag not setted, setting default True value')
            args['context_net'] = True
        if 'radius_d' not in args:
            print('WARNING: radius_d not setted, setting default value 2')
            args['radius_d'] = 2
        if 'stride' not in args:
            print('WARNING: stride not setted, setting default value 1')
            args['stride'] = 1
        if 'bulkhead' not in args:
            args['bulkhead'] = False
        return args

    def _preprocess_inputs(self, args):  ##暂时不顶替掉传统视察的输入，回头再加：line的输出
        self._left_input_batch = args['left_img']
        self._restore_shape = tf.shape(args['left_img'])[1:3]
        self._left_input_batch = tf.cast(self._left_input_batch, tf.float32)
        self._left_input_batch = preprocessing.pad_image(
            self._left_input_batch, 64)

        self._right_input_batch = args['right_img']
        self._right_input_batch = tf.cast(self._right_input_batch, tf.float32)
        self._right_input_batch = preprocessing.pad_image(
            self._right_input_batch, 64)
        self._line_input_batch = args['line_img']
        self._line_input_batch = tf.cast(self._line_input_batch, tf.float32)
        self._conf_input_batch = args['conf_img']
        self._conf_input_batch = tf.cast(self._conf_input_batch, tf.float32)

        if (args['laser_img'] != []):
            self._laser_input_batch = args['laser_img']
            self._laser_input_batch = tf.cast(self._laser_input_batch, tf.float32)
            self._laser_input_batch = preprocessing.pad_image(
                self._laser_input_batch, 64)
        if 'line_output' in args.keys():
            self._line_output_batch = args['line_output']
            self._line_output_batch = tf.cast(self._line_output_batch, tf.float32)
            self._line_output_batch = preprocessing.pad_image(
                self._line_output_batch, 64)

    def _make_disp(self, op, scale):
        op = tf.image.resize_images(tf.nn.relu(op * -20), [self._left_input_batch.get_shape()[1].value,
                                                           self._left_input_batch.get_shape()[2].value])
        op = tf.image.resize_image_with_crop_or_pad(op, self._restore_shape[0], self._restore_shape[1])
        return op

    def _stereo_estimator(self, costs, upsampled_disp=None, scope='fgc-volume-filtering'):  ##一个单层！里面6个卷积
        activation = self._leaky_relu()
        with tf.variable_scope(scope):
            # create initial cost volume
            if upsampled_disp is not None:
                volume = tf.concat([costs, upsampled_disp], -1)  ##黄色的一层
            else:
                volume = costs

            names = []  # 一个局部变量

            # disp-1
            names.append('{}/disp1'.format(scope))
            input_layer = volume
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [3, 3, volume.get_shape(
            ).as_list()[3], 128], name='disp-1', bName='biases', activation=activation))

            # disp-2:
            names.append('{}/disp2'.format(scope))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 128, 128], name='disp-2', bName='biases', activation=activation))

            # disp-3
            names.append('{}/disp3'.format(scope))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 128, 96], name='disp-3', bName='biases', activation=activation))

            # disp-4
            names.append('{}/disp4'.format(scope))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 96, 64], name='disp-4', bName='biases', activation=activation))

            # disp-5
            names.append('{}/disp5'.format(scope))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 64, 32], name='disp-5', bName='biases', activation=activation))

            # disp-6
            names.append('{}/disp6'.format(scope))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 32, 1], name='disp-6', bName='biases', activation=lambda x: x))

            return self._get_layer_as_input(names[-1])

    def _stereo_context_net(self, input, disp, last=False):
        volume = tf.concat([input, disp], -1)
        att = self._leaky_relu()
        names = []
        # context-1
        names.append('context1')
        input_layer = volume
        self._add_to_layers(names[-1], sharedLayers.dilated_conv2d(input_layer, [
            3, 3, input_layer.get_shape().as_list()[-1], 128], name='context-1', rate=1, activation=att))

        # context-2
        names.append('context2')
        input_layer = self._get_layer_as_input(names[-2])  ##防止有时候是placeholder
        self._add_to_layers(names[-1], sharedLayers.dilated_conv2d(
            input_layer, [3, 3, 128, 128], name='context-2', rate=2, activation=att))

        # context-3
        names.append('context3')
        input_layer = self._get_layer_as_input(names[-2])
        self._add_to_layers(names[-1], sharedLayers.dilated_conv2d(
            input_layer, [3, 3, 128, 128], name='context-3', rate=4, activation=att))

        # context-4
        names.append('context4')
        input_layer = self._get_layer_as_input(names[-2])
        self._add_to_layers(names[-1], sharedLayers.dilated_conv2d(
            input_layer, [3, 3, 128, 96], name='context-4', rate=8, activation=att))

        # context-5
        names.append('context5')
        input_layer = self._get_layer_as_input(names[-2])
        self._add_to_layers(names[-1], sharedLayers.dilated_conv2d(
            input_layer, [3, 3, 96, 64], name='context-5', rate=16, activation=att))

        # context-6
        names.append('context6')
        input_layer = self._get_layer_as_input(names[-2])
        self._add_to_layers(names[-1], sharedLayers.dilated_conv2d(
            input_layer, [3, 3, 64, 32], name='context-6', rate=1, activation=att))

        # context-7
        names.append('context7')
        input_layer = self._get_layer_as_input(names[-2])
        if last:
            self._add_to_layers(names[-1], sharedLayers.dilated_conv2d(
                input_layer, [3, 3, 32, 2], name='context-7', rate=1, activation=lambda x: x, batch_norm=False))
            out_2 = self._get_layer_as_input(names[-1])
            part1 = out_2[:, :, :, 0]
            part1 = tf.expand_dims(part1, axis=-1) + disp
            part2 = out_2[:, :, :, 1]
            part2 = tf.expand_dims(part2, axis=-1)
            out_2 = tf.concat([part1, part2], axis=3)
            final_disp = out_2
            self._add_to_layers('final_disp2', out_2)
            return final_disp

        else:
            self._add_to_layers(names[-1], sharedLayers.dilated_conv2d(
                input_layer, [3, 3, 32, 1], name='context-7', rate=1, activation=lambda x: x, batch_norm=False))

            final_disp = disp + self._get_layer_as_input(names[-1])
            self._add_to_layers('final_disp', final_disp)

            return final_disp

    def _pyramid_features(self, input_batch, scope='pyramid', reuse=False, layer_prefix='pyramid'):
        with tf.variable_scope(scope, reuse=reuse):
            names = []
            activation = self._leaky_relu()

            # conv1
            names.append('{}/conv1'.format(layer_prefix))
            input_layer = input_batch
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [3, 3, input_batch.get_shape(
            )[-1].value, 16], strides=2, name='conv1', bName='biases', activation=activation))

            # conv2
            names.append('{}/conv2'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])  # 拿上一次，下同
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 16, 16], strides=1, name='conv2', bName='biases', activation=activation))

            # conv3
            names.append('{}/conv3'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 16, 32], strides=2, name='conv3', bName='biases', activation=activation))

            # conv4
            names.append('{}/conv4'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 32, 32], strides=1, name='conv4', bName='biases', activation=activation))

            # conv5
            names.append('{}/conv5'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 32, 64], strides=2, name='conv5', bName='biases', activation=activation))

            # conv6
            names.append('{}/conv6'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 64, 64], strides=1, name='conv6', bName='biases', activation=activation))

            # conv7
            names.append('{}/conv7'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 64, 96], strides=2, name='conv7', bName='biases', activation=activation))

            # conv8
            names.append('{}/conv8'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 96, 96], strides=1, name='conv8', bName='biases', activation=activation))

            # conv9
            names.append('{}/conv9'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 96, 128], strides=2, name='conv9', bName='biases', activation=activation))

            # conv10
            names.append('{}/conv10'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 128, 128], strides=1, name='conv10', bName='biases', activation=activation))

            # conv11
            names.append('{}/conv11'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 128, 192], strides=2, name='conv11', bName='biases', activation=activation))

            # conv12
            names.append('{}/conv12'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 192, 192], strides=1, name='conv12', bName='biases', activation=activation))

    '''
    def _pyramid_features_laser(self, input_batch, scope='pyramid_laser', reuse=False, layer_prefix='pyramid_laser'):
        with tf.variable_scope(scope, reuse=reuse):

            names = []
            activation = self._leaky_relu()

            # conv1
            names.append('{}/conv1'.format(layer_prefix))
            input_layer = input_batch
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [3, 3, input_batch.get_shape(
            )[-1].value, 16], strides=2, name='conv1', bName='biases', activation=activation))

            # conv2
            names.append('{}/conv2'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])#拿上一次，下同
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 16, 16], strides=1, name='conv2', bName='biases', activation=activation))

            # conv3
            names.append('{}/conv3'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 16, 32], strides=2, name='conv3', bName='biases', activation=activation))

            # conv4
            names.append('{}/conv4'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 32, 32], strides=1, name='conv4', bName='biases', activation=activation))

            # conv5
            names.append('{}/conv5'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 32, 64], strides=2, name='conv5', bName='biases', activation=activation))

            # conv6
            names.append('{}/conv6'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 64, 64], strides=1, name='conv6', bName='biases', activation=activation))

            # conv7
            names.append('{}/conv7'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 64, 96], strides=2, name='conv7', bName='biases', activation=activation))

            # conv8
            names.append('{}/conv8'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 96, 96], strides=1, name='conv8', bName='biases', activation=activation))

            # conv9
            names.append('{}/conv9'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 96, 128], strides=2, name='conv9', bName='biases', activation=activation))

            # conv10
            names.append('{}/conv10'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 128, 128], strides=1, name='conv10', bName='biases', activation=activation))

            # conv11
            names.append('{}/conv11'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 128, 192], strides=2, name='conv11', bName='biases', activation=activation))

            # conv12
            names.append('{}/conv12'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                                3, 3, 192, 192], strides=1, name='conv12', bName='biases', activation=activation))
    '''

    def _build_network(self, args):
        image_height = self._left_input_batch.get_shape()[1].value
        image_width = self._left_input_batch.get_shape()[2].value
        scales = [1, 2, 4, 8, 16, 32, 64]

        #######################PYRAMID FEATURES###############################
        # setup pyramid features left
        self._pyramid_features(self._left_input_batch, scope='gc-read-pyramid', layer_prefix='left')

        # setup pyramid features right
        self._pyramid_features(self._right_input_batch, scope='gc-read-pyramid', reuse=True, layer_prefix='right')

        # self._pyramid_features_laser(self._laser_input_batch, scope='gc-read-pyramid-laser', layer_prefix='laser')
        #############################SCALE 6#################################
        with tf.variable_scope('Nconvs'):
            names = []
            num_channels = 2
            with tf.variable_scope('pre-nconv'):
                # context-1
                names.append('pre-nconv1')
                a, b = [image_height // scales[2], image_width // scales[2]]
                sz = [a, b]
                input_layer = tf.image.resize_images(self._line_input_batch, sz)
                input_conf = tf.image.resize_images(self._conf_input_batch, sz)
                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], 2 * num_channels], name='pre-nconv-1'))

                names.append('pre-nconv2')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], num_channels], name='pre-nconv-2'))

                names.append('pre-nconv3')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], num_channels], name='pre-nconv-3'))
            with tf.variable_scope('down1'):
                ds = [1, 2, 2, 1]
                names.append('down1-maxpool')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                # self._add_to_layers(names[-1], sharedLayers.nconv_max_pool(input_layer, input_conf,ksize=ds, strides = ds, name='down1-maxpool'))
                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], num_channels], strides=2, name='down1-nmaxpool'))

                names.append('down1-nconv1')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], num_channels], name='down1-nconv-1'))

                names.append('down1-nconv2')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], num_channels], name='down1-nconv-2'))  ##不变回两层
            with tf.variable_scope('down2'):
                ds = [1, 2, 2, 1]
                names.append('down2-maxpool')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                # self._add_to_layers(names[-1], sharedLayers.nconv_max_pool(input_layer, input_conf,ksize=ds, strides = ds, name='down2-maxpool'))

                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], num_channels], strides=2, name='down2-nmaxpool'))
                names.append('down2-nconv1')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], num_channels], name='down2-nconv-1'))

            with tf.variable_scope('down3'):
                ds = [1, 2, 2, 1]
                names.append('down3-maxpool')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                # self._add_to_layers(names[-1], sharedLayers.nconv_max_pool(input_layer, input_conf,ksize=ds, strides = ds, name='down3-maxpool'))

                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], num_channels], strides=2, name='down3-nmaxpool'))
                names.append('down3-nconv1')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    5, 5, input_layer.get_shape().as_list()[-1], num_channels], name='down3-nconv-1'))

            with tf.variable_scope('up1'):
                names.append('up1-nconv1')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                targ_layer, targ_conf = self._get_layer_as_input('down2-nconv1')
                input_layer = tf.image.resize_nearest_neighbor(
                    input_layer,
                    targ_layer.get_shape().as_list()[1:3],

                )
                input_conf = tf.image.resize_nearest_neighbor(
                    input_conf,
                    targ_conf.get_shape().as_list()[1:3],
                )
                input_conf = tf.concat([input_conf, targ_conf], axis=3)
                input_layer = tf.concat([input_layer, targ_layer], axis=3)

                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    3, 3, input_layer.get_shape().as_list()[-1], num_channels], name='up1-nconv-1'))

            with tf.variable_scope('up2'):
                names.append('up2-nconv1')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                targ_layer, targ_conf = self._get_layer_as_input('down1-nconv2')
                input_layer = tf.image.resize_nearest_neighbor(
                    input_layer,
                    targ_layer.get_shape().as_list()[1:3],

                )
                input_conf = tf.image.resize_nearest_neighbor(
                    input_conf,
                    targ_conf.get_shape().as_list()[1:3],
                )
                input_conf = tf.concat([input_conf, targ_conf], axis=3)
                input_layer = tf.concat([input_layer, targ_layer], axis=3)

                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    3, 3, input_layer.get_shape().as_list()[-1], num_channels], name='up2-nconv-1'))

            with tf.variable_scope('up3'):
                names.append('up3-nconv1')
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                targ_layer, targ_conf = self._get_layer_as_input('pre-nconv3')
                input_layer = tf.image.resize_nearest_neighbor(
                    input_layer,
                    targ_layer.get_shape().as_list()[1:3],

                )
                input_conf = tf.image.resize_nearest_neighbor(
                    input_conf,
                    targ_conf.get_shape().as_list()[1:3],
                )
                input_conf = tf.concat([input_conf, targ_conf], axis=3)
                input_layer = tf.concat([input_layer, targ_layer], axis=3)

                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    3, 3, input_layer.get_shape().as_list()[-1], num_channels], name='up3-nconv-1'))

                names.append('final-nconv')  ##出来的算个loss
                input_layer, input_conf = self._get_layer_as_input(names[-2])
                '''
                a,b = self._layers['final_disp3'].get_shape()[1].value,self._layers['final_disp3'].get_shape()[2].value
                sz = [a,b]
                input_layer = tf.image.resize_nearest_neighbor(input_layer,
                                                             sz)
                input_conf = tf.image.resize_nearest_neighbor(input_conf,sz)
                '''
                self._add_to_layers(names[-1], sharedLayers.nconv2d(input_layer, input_conf, [
                    1, 1, input_layer.get_shape().as_list()[-1], 1], name='up3-nconv-2'))
                n_layer, n_conf = self._get_layer_as_input(names[-1])
                rescaled_prediction = tf.image.resize_images(n_layer,
                                                             [image_height, image_width]) * -20.
                self.my_disparities.append(rescaled_prediction)
                self.output_laser_conf = tf.image.resize_images(n_conf,
                                                                [image_height, image_width])  ##for loss

                # self.fordisp_line = tf.image.resize_images(n_layer,[????])

        with tf.variable_scope("G6"):
            with tf.variable_scope("unary-6"):
                left_0_sample_6 = self._get_layer_as_input('left/conv12')
                right_0_sample_6 = self._get_layer_as_input('right/conv12')
                # laser, conf = self._get_layer_as_input('final-nconv')#--------------------，也不知道应该加哪一层
                # laser = tf.concat([laser, conf], axis=-1)
            with tf.variable_scope("fgc-volume-creator-6"):
                dsi_6 = self._stereo_cost_volume_correlation(left_0_sample_6, right_0_sample_6, args['radius_d'],
                                                             args['stride'])

            V6 = self._stereo_estimator(dsi_6, scope="fgc-volume-filtering-6")
            real_disp_v6 = self._make_disp(V6, scales[6])
            self._disparities.append(real_disp_v6)
            u5 = tf.image.resize_images(V6, [image_height // scales[5], image_width // scales[5]]) * 20. / scales[5]
            if args['bulkhead']:
                u5 = tf.stop_gradient(u5)

        ############################SCALE 5###################################
        with tf.variable_scope("G5"):
            with tf.variable_scope("unary-5"):
                left_0_sample_5 = self._get_layer_as_input('left/conv10')
                # line_0_sample_5,conf = self._get_layer_as_input('down3-nconv1')
                # line_0_sample_5 = tf.concat([line_0_sample_5,conf],axis = -1)
                if args['warping']:
                    right_0_sample_5 = self._linear_warping(self._get_layer_as_input('right/conv10'),
                                                            self._build_indeces(tf.concat([u5, tf.zeros_like(u5)], -1)))
                else:
                    right_0_sample_5 = self._get_layer_as_input('right/conv10')

            with tf.variable_scope("fgc-volume-creator-5"):
                dsi_5 = self._stereo_cost_volume_correlation(left_0_sample_5, right_0_sample_5, args['radius_d'],
                                                             args['stride'])

            V5 = self._stereo_estimator(dsi_5, upsampled_disp=u5, scope="fgc-volume-filtering-5")
            real_disp_v5 = self._make_disp(V5, scales[5])
            self._disparities.append(real_disp_v5)
            u4 = tf.image.resize_images(V5, [image_height // scales[4], image_width // scales[4]]) * 20. / scales[4]
            if args['bulkhead']:
                u4 = tf.stop_gradient(u4)

        ############################SCALE 4###################################
        with tf.variable_scope('G4'):
            with tf.variable_scope('unary-4'):
                left_0_sample_4 = self._get_layer_as_input('left/conv8')
                line_0_sample_4, conf = self._get_layer_as_input('up1-nconv1')
                line_0_sample_4 = tf.concat([line_0_sample_4, conf], axis=-1)
                if args['warping']:
                    right_0_sample_4 = self._linear_warping(self._get_layer_as_input('right/conv8'),
                                                            self._build_indeces(tf.concat([u4, tf.zeros_like(u4)], -1)))
                else:
                    right_0_sample_4 = self._get_layer_as_input('right/conv8')

            with tf.variable_scope("fgc-volume-creator-4"):
                dsi_4 = self._stereo_cost_volume_correlation_laser(left_0_sample_4, right_0_sample_4, args['radius_d'],
                                                                   line_0_sample_4, args['stride'])

            V4 = self._stereo_estimator(dsi_4, upsampled_disp=u4, scope="fgc-volume-filtering-4")
            real_disp_v4 = self._make_disp(V4, scales[4])
            self._disparities.append(real_disp_v4)
            u3 = tf.image.resize_images(V4, [image_height // scales[3], image_width // scales[3]]) * 20. / scales[3]
            if args['bulkhead']:
                u3 = tf.stop_gradient(u3)

        ############################SCALE 3###################################
        with tf.variable_scope('G3'):
            with tf.variable_scope('unary-3'):
                left_0_sample_3 = self._get_layer_as_input('left/conv6')
                line_0_sample_3, conf = self._get_layer_as_input('up2-nconv1')
                line_0_sample_3 = tf.concat([line_0_sample_3, conf], axis=-1)
                if args['warping']:
                    right_0_sample_3 = self._linear_warping(self._get_layer_as_input('right/conv6'),
                                                            self._build_indeces(tf.concat([u3, tf.zeros_like(u3)], -1)))
                else:
                    right_0_sample_3 = self._get_layer_as_input('right/conv6')

            with tf.variable_scope("fgc-volume-creator-3"):
                dsi_3 = self._stereo_cost_volume_correlation_laser(left_0_sample_3, right_0_sample_3, args['radius_d'],
                                                                   line_0_sample_3, args['stride'])

            V3 = self._stereo_estimator(dsi_3, upsampled_disp=u3, scope="fgc-volume-filtering-3")
            real_disp_v3 = self._make_disp(V3, scales[3])
            self._disparities.append(real_disp_v3)
            u2 = tf.image.resize_images(V3, [image_height // scales[2], image_width // scales[2]]) * 20. / scales[2]
            if args['bulkhead']:
                u2 = tf.stop_gradient(u2)

        #################################SCALE 2###############################
        with tf.variable_scope('G2'):
            with tf.variable_scope('unary-2'):
                left_0_sample_2 = self._get_layer_as_input('left/conv4')
                line_0_sample_2, conf = self._get_layer_as_input('final-nconv')
                line_0_sample_2 = tf.concat([line_0_sample_2, conf], axis=-1)
                if args['warping']:
                    right_0_sample_2 = self._linear_warping(self._get_layer_as_input('right/conv4'),
                                                            self._build_indeces(tf.concat([u2, tf.zeros_like(u2)], -1)))
                else:
                    right_0_sample_2 = self._get_layer_as_input('right/conv4')

            with tf.variable_scope("fgc-volume-creator-2"):
                dsi_2 = self._stereo_cost_volume_correlation_laser(left_0_sample_2, right_0_sample_2, args['radius_d'],
                                                                   line_0_sample_2, args['stride'])

            self._stereo_estimator(dsi_2, upsampled_disp=u2, scope="fgc-volume-filtering-2")
            V2_init = self._get_layer_as_input('fgc-volume-filtering-2/disp6')
            ##################################################
            # real_disp_v2 = self._make_disp(V2_init,scales[2])
            # self._disparities.append(real_disp_v2)

        if args['context_net']:
            V2 = self._stereo_context_net(left_0_sample_2, V2_init, last=True)
            real_disp3_v2_context = self._make_disp(V2, scales[2])  # 出来有padding64的
            self._disparities.append(real_disp3_v2_context[:, :, :, 0])
            # 先不加addtolayers了，加的一定需要调节权重，回头加
        else:  ##never
            V2 = V2_init
            self._add_to_layers('final_disp', V2)
            self._disparities.append(real_disp_v2)

        rescaled_prediction = tf.image.resize_images(tf.expand_dims(self._layers['final_disp2'][:, :, :, 0], axis=-1),
                                                     [image_height, image_width]) * -20.  # 如果不加维数会出现奇怪的效果
        self._layers['rescaled_prediction'] = tf.image.resize_image_with_crop_or_pad(rescaled_prediction,
                                                                                     self._restore_shape[0],
                                                                                     self._restore_shape[1])
        self._disp_conf = tf.expand_dims(self._layers['final_disp2'][:, :, :, 1], axis=-1)
        # self._disp_guide = tf.expand_dims(self._layers['final_disp3'][:, :, :, 2],axis = -1)

        self._disparities.append(self._layers['rescaled_prediction'])  ###最后被fetch

        # 最后几层省掉？？？
        with tf.variable_scope('fusion'):
            n_layer, n_conf = self._get_layer_as_input('final-nconv')

            a, b = self._layers['final_disp2'].get_shape()[1].value, self._layers['final_disp2'].get_shape()[2].value
            sz = [a, b]
            n_layer = tf.image.resize_images(n_layer,
                                             sz)
            n_conf = tf.image.resize_images(n_conf, sz)

            conf = tf.stack([n_conf, self._disp_conf], axis=0)
            disp = tf.stack([n_layer, tf.expand_dims(self._layers['final_disp2'][:, :, :, 0], axis=-1)], axis=0)
            conf = tf.nn.softmax(
                conf,
                axis=0
            )
            _disp = tf.multiply(disp, conf)
            weighted_disp = _disp[0, :, :, :, :] + _disp[1, :, :, :, :]

            volume = tf.concat(
                [n_layer, n_conf, weighted_disp, tf.expand_dims(self._layers['final_disp2'][:, :, :, 0], axis=-1),
                 self._disp_conf], axis=-1)

            activation = self._leaky_relu()

            scope = 'fusion'
            # disp-1
            names.append('{}/fu1'.format(scope))
            input_layer = volume
            self._add_to_layers(names[-1],
                                sharedLayers.conv2d(input_layer, [3, 3, input_layer.get_shape().as_list()[3], 16],
                                                    strides=2, name='fu-1', bName='biases', activation=activation))
            # fu-2:
            '''self._left_input_batch = preprocessing.pad_image(
            self._left_input_batch, 4) if 不是整数倍'''
            names.append('{}/fu2'.format(scope))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 16, 8], strides=1, name='fu-2', bName='biases', activation=activation))

            # fu-3
            names.append('{}/fu3'.format(scope))
            input_layer = self._get_layer_as_input(names[-2])
            # input_layer = tf.concat([input_layer, self._get_layer_as_input('{}/fu1'.format(scope))], axis=3)
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, 8, 4], name='fu-3', bName='biases', activation=activation))

            # fu-4:
            names.append('{}/fu4'.format(scope))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d_transpose(input_layer, [
                3, 3, 4, 4], strides=2, name='fu-4', bName='biases', activation=activation))

            # fu-5
            names.append('{}/fu5'.format(scope))
            input_layer = self._get_layer_as_input(names[-2])
            input_layer = tf.concat([input_layer, volume], axis=3)
            print(input_layer.shape)  # (2, 20, 60, 8)  注意反向conv参数要换位置？？？以及这里死活加不上layer，只能transpose
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [
                3, 3, input_layer.get_shape().as_list()[3], 1], name='fu-5', bName='biases', activation=lambda
                x: x))  ###!!!![height, width, output_channels, in_channels]
            # 换成了加法。
            rescaled_prediction = tf.image.resize_images(
                self._get_layer_as_input(names[-1]) + tf.expand_dims(self._layers['final_disp2'][:, :, :, 0], axis=-1),
                [image_height, image_width]) * -20.
            self.my_disparities.append(rescaled_prediction)

    def _leaky_relu(self):
        return lambda x: tf.maximum(0.2 * x, x)

    # Utility functions
    def _stereo_cost_volume_correlation(self, reference, target, radius_x, stride=1):

        cost_curve = sharedLayers.correlation(reference, target, radius_x, stride=stride)
        cost_curve = tf.concat([reference, cost_curve], axis=3)  # channel上

        return cost_curve

    def _stereo_cost_volume_correlation_laser(self, reference, target, radius_x, laser, stride=1):
        laser = tf.image.resize_images(laser,
                                       [reference.get_shape().as_list()[1], reference.get_shape().as_list()[2]])
        cost_curve = sharedLayers.correlation(reference, target, radius_x, stride=stride)
        cost_curve = tf.concat([reference, cost_curve, laser], axis=3)  # channel上

        return cost_curve

    def _build_indeces(self, coords):
        batches = coords.get_shape().as_list()[0]

        height = coords.get_shape().as_list()[1]
        width = coords.get_shape().as_list()[2]
        pixel_coords = np.ones((1, height, width, 2))
        batches_coords = np.ones((batches, height, width, 1))

        for i in range(0, batches):
            batches_coords[i][:][:][:] = i
        # build pixel coordinates and their disparity
        for i in range(0, height):
            for j in range(0, width):
                pixel_coords[0][i][j][0] = j
                pixel_coords[0][i][j][1] = i  ##原来的坐标，一会直接加上偏移

        pixel_coords = tf.constant(pixel_coords, tf.float32)
        output = tf.concat([batches_coords, pixel_coords + coords], -1)

        return output

    def _linear_warping(self, imgs, coords):
        shape = coords.get_shape().as_list()

        # coords = tf.reshape(coords, [shape[1], shape[2], shape[0], shape[3]])
        coord_b, coords_x, coords_y = tf.split(coords, [1, 1, 1], axis=3)

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype=tf.float32)

        x0_safe = tf.clip_by_value(x0, zero[0], x_max)
        y0_safe = tf.clip_by_value(y0, zero[0], y_max)
        x1_safe = tf.clip_by_value(x1, zero[0], x_max)

        # bilinear interp weights, with points outside the grid having weight 0  x0x1就差1
        wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')  # cast后结果只有越界的点才和原来不同，equal是0，否则是1
        wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')

        # print(x0_safe.get_shape().as_list())

        im00 = tf.cast(tf.gather_nd(imgs, tf.cast(
            tf.concat([coord_b, y0_safe, x0_safe], -1), 'int32')), 'float32')  ##gathernd就是取，后面的这些三维的坐标
        im01 = tf.cast(tf.gather_nd(imgs, tf.cast(
            tf.concat([coord_b, y0_safe, x1_safe], -1), 'int32')), 'float32')

        output = tf.add_n([
            wt_x0 * im00, wt_x1 * im01
        ])

        return output
