import tensorflow as tf
import numpy as np
import argparse
import Nets
import os
import sys
import time
import cv2
import json
import datetime
import shutil
from matplotlib import pyplot as plt
from Data_utils import data_reader_norinosgbm, weights_utils, preprocessing
from Losses import loss_factory
from Sampler import sampler_factory

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# static params
PIXEL_TH = 3
MAX_DISP = 192


def main(args):
    # read input data
    with tf.name_scope('input_reader'):
        with tf.name_scope('training_set_reader'):
            data_set = data_reader_norinosgbm.dataset(
                args.trainingSet,
                batch_size=args.batchSize,
                crop_shape=args.imageShape,
                num_epochs=args.numEpochs,
                augment=args.augment,
                is_training=True,
                shuffle=True
            )
            left_img_batch, right_img_batch, gt_image_batch, laser_image_batch, line_img_batch = data_set.get_batch()
            conf_img_batch = tf.where(tf.greater(line_img_batch, 0), \
                                      tf.ones_like(line_img_batch, dtype=tf.float32), \
                                      tf.zeros_like(line_img_batch, dtype=tf.float32))
            inputs = {
                'left': left_img_batch,
                'right': right_img_batch,
                'target': gt_image_batch,
                'laser': laser_image_batch,
                'line': line_img_batch,
                'conf': conf_img_batch
            }
        if args.validationSet is not None:
            with tf.name_scope('validation_set_reader'):
                validation_set = data_reader_norinosgbm.dataset(
                    args.validationSet,
                    batch_size=args.batchSize,
                    augment=False,
                    is_training=False,
                    shuffle=True
                )
                left_val_batch, right_val_batch, gt_val_batch, laser_val_batch, line_val_batch = validation_set.get_batch()  # print(left_val_batch.shape, right_val_batch.shape)# build network
                conf_val_batch = tf.where(tf.greater(line_val_batch, 0), \
                                          tf.ones_like(line_val_batch, dtype=tf.float32), \
                                          tf.zeros_like(line_val_batch, dtype=tf.float32))
    with tf.variable_scope('model') as scope:
        net_args = {}
        net_args['left_img'] = left_img_batch
        net_args['right_img'] = right_img_batch
        net_args['laser_img'] = laser_image_batch
        net_args['line_img'] = line_img_batch
        net_args['conf_img'] = conf_img_batch
        net_args['split_layers'] = [None]
        net_args['sequence'] = True
        net_args['train_portion'] = 'BEGIN'
        net_args['bulkhead'] = False
        stereo_net = Nets.get_stereo_net(args.modelName, net_args)
        print('Stereo Prediction Model:\n', stereo_net)
        predictions = stereo_net.get_disparities()
        # my_predictions = stereo_net.get_my_disparities()

        # full_res_disp_prev = predictions[-3]
        # full_res_disp = predictions[-2]
        # full_res_disp_inserted = predictions[-1]

        full_res_disp = predictions[-1]

        # predictions+=my_predictions
        # output_laser_conf=stereo_net.get_laser_conf()

        if args.validationSet is not None:
            scope.reuse_variables()
            net_args['left_img'] = left_val_batch
            net_args['right_img'] = right_val_batch
            net_args['laser_img'] = laser_val_batch
            net_args['line_img'] = line_val_batch
            net_args['conf_img'] = conf_val_batch
            val_stereo_net = Nets.get_stereo_net(args.modelName, net_args)

            val_prevprediction = val_stereo_net.get_disparities()[-1]
            # val_prediction = val_stereo_net.get_disparities()[-2]
            # val_prevprediction = val_stereo_net.get_disparities()[-3]
            # inserteds = val_stereo_net.get_final_disp_inserteds()
            # val_prevprediction4 = val_stereo_net.get_disparities()[-5]
            # val_prevprediction2 = val_stereo_net.get_disparities()[-7]


            # val_prediction = val_stereo_net.get_my_disparities()[-1]
            # val_nconvprediction = val_stereo_net.get_my_disparities()[0]

    if args.validationSet is not None:
        with tf.variable_scope('validation_error'):
            # compute error against gt
            # abs_err = tf.abs(val_prediction - gt_val_batch)/gt_val_batch


            # abs_err = tf.abs(val_prediction - gt_val_batch)
            abs_err_prev = tf.abs(val_prevprediction - gt_val_batch)

            # abs_err_inserted = tf.abs(val_prediction_inserted - gt_val_batch)
            # abs_errs = [tf.abs(val_prediction_inserted - gt_val_batch) for val_prediction_inserted in inserteds]

            # abs_err_ori = tf.abs(laser_val_batch/16 - gt_val_batch)
            valid_map = tf.where(tf.equal(gt_val_batch, 0), tf.zeros_like(gt_val_batch, dtype=tf.float32),
                                 tf.ones_like(gt_val_batch, dtype=tf.float32))
            valid_map_ori = valid_map * tf.where(tf.equal(laser_val_batch, 0),
                                                 tf.zeros_like(laser_val_batch, dtype=tf.float32),
                                                 tf.ones_like(laser_val_batch, dtype=tf.float32))
            # filtered_error = abs_err * valid_map
            filtered_error_prev = abs_err_prev * valid_map
            # filtered_error_inserted = abs_err_inserted * valid_map
            # filtered_errors = [inserted * valid_map for inserted in abs_errs]
            # filtered_error_ori = abs_err_ori * valid_map_ori

            # EPE = tf.reduce_sum(filtered_error) / tf.reduce_sum(valid_map)
            EPE_prev = tf.reduce_sum(filtered_error_prev) / tf.reduce_sum(valid_map)
            # EPE_inserted = tf.reduce_sum(filtered_error_inserted) / tf.reduce_sum(valid_map)
            # EPE_inserteds = [tf.reduce_sum(filtered_error_inserted) / tf.reduce_sum(valid_map) for
            #                  filtered_error_inserted in filtered_errors]

            # EPE_ori = tf.reduce_sum(filtered_error_ori) / tf.reduce_sum(valid_map_ori)

            # bad_5_perc = tf.where(tf.greater(tf.abs(filtered_error / (gt_val_batch + 1.0 - valid_map)), 0.05),
            #                       tf.ones_like(abs_err, dtype=tf.float32),
            #                       tf.zeros_like(abs_err, dtype=tf.float32))
            # bad_pixel_abs = tf.where(tf.greater(filtered_error, PIXEL_TH),
            #                          tf.ones_like(filtered_error, dtype=tf.float32),
            #                          tf.zeros_like(filtered_error, dtype=tf.float32))
            # bad_pixel_perc = tf.reduce_sum(bad_pixel_abs * bad_5_perc) / tf.reduce_sum(valid_map)
            bad_5_perc_prev = tf.where(tf.greater(tf.abs(filtered_error_prev / (gt_val_batch + 1.0 - valid_map)), 0.05),
                                       tf.ones_like(abs_err_prev, dtype=tf.float32),
                                       tf.zeros_like(abs_err_prev, dtype=tf.float32))
            bad_pixel_abs_prev = tf.where(tf.greater(filtered_error_prev, PIXEL_TH),
                                          tf.ones_like(filtered_error_prev, dtype=tf.float32),
                                          tf.zeros_like(filtered_error_prev, dtype=tf.float32))
            bad_pixel_perc_prev = tf.reduce_sum(bad_pixel_abs_prev * bad_5_perc_prev) / tf.reduce_sum(valid_map)

            # bad_5_perc_inserted = tf.where(
            #     tf.greater(tf.abs(filtered_error_inserted / (gt_val_batch + 1.0 - valid_map)), 0.05),
            #     tf.ones_like(abs_err, dtype=tf.float32),
            #     tf.zeros_like(abs_err, dtype=tf.float32))
            # bad_pixel_abs_inserted = tf.where(tf.greater(filtered_error_inserted, PIXEL_TH),
            #                                   tf.ones_like(filtered_error, dtype=tf.float32),
            #                                   tf.zeros_like(filtered_error, dtype=tf.float32))
            # bad_pixel_perc_inserted = tf.reduce_sum(bad_pixel_abs_inserted * bad_5_perc_inserted) / tf.reduce_sum(
            #     valid_map)
            #
            # bad_5_perc_inserteds = [tf.where(tf.greater(tf.abs(finserted / (gt_val_batch + 1.0 - valid_map)), 0.05),
            #                                  tf.ones_like(abs_err, dtype=tf.float32),
            #                                  tf.zeros_like(abs_err, dtype=tf.float32)) for finserted in filtered_errors]
            # bad_pixel_abs_inserteds = [tf.where(tf.greater(finserted, PIXEL_TH),
            #                                     tf.ones_like(filtered_error, dtype=tf.float32),
            #                                     tf.zeros_like(filtered_error, dtype=tf.float32)) for finserted in
            #                            filtered_errors]
            # bad_pixel_perc_inserteds = [tf.reduce_sum(bad_pixel_abs_i * bad_5_perc_i) / tf.reduce_sum(valid_map) for
            #                             bad_pixel_abs_i, bad_5_perc_i in
            #                             zip(bad_5_perc_inserteds, bad_pixel_abs_inserteds)]
            '''
            bad_5_perc_ori = tf.where(tf.greater(tf.abs(filtered_error_ori/(gt_val_batch+ 1.0 - valid_map)), 0.05),
                                      tf.ones_like(abs_err_ori, dtype=tf.float32),
                                      tf.zeros_like(abs_err_ori, dtype=tf.float32))
            bad_pixel_abs_ori = tf.where(tf.greater(filtered_error_ori, PIXEL_TH),
                                         tf.ones_like(filtered_error_ori, dtype=tf.float32),
                                         tf.zeros_like(filtered_error_ori, dtype=tf.float32))
            bad_pixel_perc_ori = tf.reduce_sum(bad_pixel_abs_ori * bad_5_perc_ori) / tf.reduce_sum(valid_map_ori)
            '''
            r_prev = tf.abs(val_prevprediction - gt_val_batch) ** 2
            rmse_prev = r_prev * valid_map
            # rmse = np.sqrt(rmse.mean())
            rm_prev = tf.sqrt(tf.reduce_sum(rmse_prev) / tf.reduce_sum(valid_map))

            # r_inserted = tf.abs(val_prediction_inserted - gt_val_batch) ** 2
            # rmse_inserted = r_inserted * valid_map
            # # rmse = np.sqrt(rmse.mean())
            # rm_inserted = tf.sqrt(tf.reduce_sum(rmse_inserted) / tf.reduce_sum(valid_map))
            #
            # r = tf.abs(val_prediction - gt_val_batch) ** 2
            # rmse = r * valid_map
            # # rmse = np.sqrt(rmse.mean())
            # rm = tf.sqrt(tf.reduce_sum(rmse) / tf.reduce_sum(valid_map))




            # delta_err = tf.maximum(val_prediction / (gt_val_batch + 1.0 - valid_map),
            #                        gt_val_batch / (val_prediction + 1.0 - valid_map))
            # delta_err = valid_map * delta_err
            # delta1 = tf.where(tf.less(tf.abs(delta_err), 1.25),
            #                   tf.ones_like(filtered_error, dtype=tf.float32),
            #                   tf.zeros_like(filtered_error, dtype=tf.float32))
            # delta1 = valid_map * delta1
            # delta1 = tf.reduce_sum(delta1) / tf.reduce_sum(valid_map)
            # delta2 = tf.where(tf.less(tf.abs(delta_err), 1.25 * 1.25),
            #                   tf.ones_like(filtered_error, dtype=tf.float32),
            #                   tf.zeros_like(filtered_error, dtype=tf.float32))
            # delta2 = valid_map * delta2
            # delta2 = tf.reduce_sum(delta2) / tf.reduce_sum(valid_map)




            def gradient_x(img):
                # Pad input to keep output size consistent
                # img = tf.pad(img, [0, 1, 0, 0], mode="SYMMETRIC")
                img = tf.pad(img, [[0, 0], [0, 0], [0, 1], [0, 0]], mode="SYMMETRIC")

                gx = img[:, :, :-1, :] - img[:, :, 1:, :]  # NHWC in tf
                return gx

            def gradient_y(img):
                # Pad input to keep output size consistent
                img = tf.pad(img, [[0, 0], [0, 1], [0, 0], [0, 0]], mode="SYMMETRIC")
                gy = img[:, :-1, :, :] - img[:, 1:, :, :]  # NHWC
                return gy

            sum_grad = tf.cast(tf.expand_dims(tf.reduce_mean(gradient_x(left_val_batch) + gradient_y(left_val_batch), 3), 3), tf.float32)
            mean_grad = tf.metrics.mean(sum_grad)[0]
            # print(mean_grad)
            mask_grad = tf.where(tf.greater(sum_grad, mean_grad),
                                 tf.ones_like(abs_err_prev, dtype=tf.float32),
                                 tf.zeros_like(abs_err_prev, dtype=tf.float32))
            bad_pixel_perc_grad = tf.reduce_sum(bad_pixel_abs_prev * bad_5_perc_prev * mask_grad * valid_map) / tf.reduce_sum(
                valid_map * mask_grad)
            EPE_grad = tf.reduce_sum(filtered_error_prev * mask_grad) / tf.reduce_sum(mask_grad)

            # tf.summary.scalar('abs_err', EPE)
            tf.summary.scalar('abs_err_prev', EPE_prev)
            # tf.summary.scalar('abs_err_inserted', EPE_inserted)
            tf.summary.scalar('abs_err_grad', EPE_grad)
            # for i in range(len(EPE_inserteds)):
            #     tf.summary.scalar('EPE_err_inserted' + str(i), EPE_inserteds[i])
            # tf.summary.scalar('rmse', rm)
            tf.summary.scalar('rmse_prev', rm_prev)
            # tf.summary.scalar('rmse_inserted', rm_inserted)
            # tf.summary.scalar('bad3', bad_pixel_perc)
            tf.summary.scalar('bad3_grad', bad_pixel_perc_grad)
            tf.summary.scalar('bad3_prev', bad_pixel_perc_prev)
            # tf.summary.scalar('bad3_inserted', bad_pixel_perc_inserted)
            # for i in range(len(EPE_inserteds)):
            #     tf.summary.scalar('bad3_inserted' + str(i), bad_pixel_perc_inserteds[i])
            #
            # tf.summary.scalar('delta1', delta1)
            # tf.summary.scalar('delta2', delta2)
            tf.summary.image('left_img', left_val_batch, max_outputs=1)
            # tf.summary.image('val_prediction', preprocessing.colorize_img(val_prediction, cmap='jet'), max_outputs=1)
            tf.summary.image('grad_mask', preprocessing.colorize_img(mask_grad, cmap='jet'), max_outputs=1)
            # tf.summary.image('val_prevprediction2', preprocessing.colorize_img(val_prevprediction2, cmap='jet'),
            #                  max_outputs=1)
            # tf.summary.image('val_prevprediction4', preprocessing.colorize_img(val_prevprediction4, cmap='jet'),
            #                  max_outputs=1)
            tf.summary.image('val_prevprediction', preprocessing.colorize_img(val_prevprediction, cmap='jet'),
                             max_outputs=1)
            # tf.summary.image('val_prediction_inserted', preprocessing.colorize_img(val_prediction_inserted, cmap='jet'),
            #                  max_outputs=1)
            tf.summary.image('val_gt', preprocessing.colorize_img(gt_val_batch, cmap='jet'), max_outputs=1)
            # tf.summary.image('conf_ret', preprocessing.colorize_img(conf_prediction, cmap='jet'), max_outputs=1)

    with tf.name_scope('training_error'):
        # build train ops
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(args.lr, global_step, args.decayStep, 0.5, staircase=True)
        disparity_trainer = tf.train.AdamOptimizer(args.lr, 0.9)

        # l1 regression loss for each scale mutiplied by the corresponding weight
        assert (len(args.lossWeights) == len(predictions)), str(len(predictions))  ####怎么续
        full_reconstruction_loss = loss_factory.get_supervised_loss(args.lossType, multiScale=True, logs=False,
                                                                    weights=args.lossWeights, max_disp=MAX_DISP)(predictions, inputs, None)
        # full_reconstruction_loss=loss_factory.get_reprojection_loss('mean_SSIM_l1',multiScale=True,logs=False,
        # weights=args.lossWeights,reduced=True)(predictions,inputs)

        # add summaries
        tf.summary.image('full_res_disp', preprocessing.colorize_img(full_res_disp, cmap='jet'), max_outputs=1)
        tf.summary.image('gt_disp', preprocessing.colorize_img(gt_image_batch, cmap='jet'), max_outputs=1)
        tf.summary.scalar('full_reconstruction_loss', full_reconstruction_loss)

    # create summary logger
    summary_op = tf.summary.merge_all()
    logger = tf.summary.FileWriter(args.output)

    # create saver
    main_saver = tf.train.Saver(max_to_keep=3)

    # start session
    gpu_options = tf.GPUOptions(allow_growth=True)
    max_steps = data_set.get_max_steps()
    exec_time = 0
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # restore disparity inference weights
        restored, step_eval, vars_to_restore = weights_utils.check_for_weights_or_restore_them(args.output, sess,
                                                                                               initial_weights=args.weights,
                                                                                               ignore_list=['conv00'])  # ['context-','laser', 'G6','nograd'])
        print('Disparity Net Restored?: {} from step {}'.format(restored, step_eval))
        step_eval = 0
        ###vars_to_restore=[]
        # 正好如果继续训练很安全，全部都可以训了
        first_train_vars = [v for v in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)]  # if 'nograd' not in v.name and 'left' not in v.name and 'right' not in v.name]#v.name[:-2] not in vars_to_restore and
        # train_op = disparity_trainer.minimize(full_reconstruction_loss, global_step=global_step,
        # var_list=first_train_vars)
        # clipping
        gvs = disparity_trainer.compute_gradients(full_reconstruction_loss, first_train_vars)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
        # capped_gvs = gvs
        train_op = disparity_trainer.apply_gradients(capped_gvs, global_step=global_step)
        # init stuff
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])  # bixuzaizuihou
        # will wipe out the restored weights
        if len(vars_to_restore) > 0:
            restorer = tf.train.Saver(var_list=vars_to_restore)
            # initial_weights = tf.train.latest_checkpoint(),xiamian initial
            restorer.restore(sess, args.weights)
        sess.run(global_step.assign(step_eval))
        try:
            start_time = time.time()
            while True:
                tf_fetches = [global_step, train_op, full_reconstruction_loss, val_prevprediction]

                if step_eval % 100 == 0:
                    # summaries
                    tf_fetches = tf_fetches + [summary_op]

                # run network
                run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
                fetches = sess.run(tf_fetches, options=run_options)  # 全跑一遍

                if step_eval % 100 == 0:
                    # log on terminal
                    fbTime = (time.time() - start_time)
                    exec_time += fbTime
                    fbTime = fbTime / 100
                    logger.add_summary(fetches[-1], global_step=step_eval)
                    missing_time = (max_steps - step_eval) * fbTime
                    print(
                        'Step:{:4d}\tLoss:{:.2f}\tf/b time:{:3f}\tMissing time:{}'.format(step_eval, fetches[2], fbTime,
                                                                                          datetime.timedelta(
                                                                                              seconds=missing_time)))
                    start_time = time.time()

                if step_eval % 10000 == 0:
                    ckpt = os.path.join(args.output, 'weights.ckpt')
                    main_saver.save(sess, ckpt, global_step=step_eval)

                step_eval = fetches[0]
        except tf.errors.OutOfRangeError:
            pass
        finally:
            print('All Done, Bye Bye!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training of a Deep Stereo Network')
    parser.add_argument("--trainingSet", help='path to the list file with training set',
                        default='s3://bucketjy/nori_flying_train.list',
                        type=str)  ############
    parser.add_argument("--validationSet", help="path to the list file with the validation set",
                        default='s3://bucketjy/nori_flying_test.list', type=str)  #
    parser.add_argument("-o", "--output", help="path to the output folder where the results will be saved",
                        default='respsmn/', type=str)
    parser.add_argument("--weights", help="path to the initial weights for the disparity estimation network (OPTIONAL)", default = None)  # ../realtimelaser/reslas/')
    parser.add_argument("--modelName", help="name of the stereo model to be used", default="psmnet",
                        choices=Nets.STEREO_FACTORY.keys())
    parser.add_argument("--lr", help="initial value for learning rate", default=0.0001, type=float)  # 1
    parser.add_argument("--imageShape", help='two int for image shape [height,width]', nargs='+', type=int,
                        default=[320, 960])  # 752x480
    parser.add_argument("--batchSize", help='batch size to use during training', type=int, default=1)
    parser.add_argument("--numEpochs", help='number of training epochs', type=int, default=200)
    parser.add_argument("--augment", help="flag to enable data augmentation", default=True, action='store_true')
    parser.add_argument("--lossWeights", help="weights for loss at different resolution from full to lower res",
                        nargs='+', default=[1.0], type=float)
    parser.add_argument('--lossType', help="Type of supervised loss to use",
                        choices=loss_factory.SUPERVISED_LOSS.keys(), default="mean_l1", type=str)
    parser.add_argument("--decayStep", help="halve learning rate after this many steps", type=int, default=500000)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, 'params.sh'), 'w+') as out:
        sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
        out.write('#!/bin/bash\n')
        out.write('python3 ')
        out.write(' '.join(sys.argv))
        out.write('\n')
    main(args)

