# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import imagenet
from nets import GoogLeNet
from preprocessing import preprocessing_factory
import time
import input_data
import os
import errno
from six.moves import xrange
import numpy as np
import kmeans
import matlab.engine


slim = tf.contrib.slim

from tf_flags import FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
        num_samples_per_epoch: The number of samples in each epoch of training.
        global_step: The global_step tensor.
    Returns:
        A `Tensor` representing the learning rate.
    Raises:
        ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
        learning_rate: A scalar or `Tensor` learning rate.
    Returns:
        An instance of an optimizer.
    Raises:
        ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate,
                                               rho=FLAGS.adadelta_rho,
                                               epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate,
                                              initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate,
                                           beta1=FLAGS.adam_beta1,
                                           beta2=FLAGS.adam_beta2,
                                           epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate,
                                           learning_rate_power=FLAGS.ftrl_learning_rate_power,
                                           initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
                                           l1_regularization_strength=FLAGS.ftrl_l1,
                                           l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum=FLAGS.momentum,
                                               name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                              decay=FLAGS.rmsprop_decay,
                                              momentum=FLAGS.rmsprop_momentum,
                                              epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)

    return optimizer


def accuracy(logits, labels):
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def tower_loss_xentropy_dense(logits, labels):
    labels = tf.cast(labels, tf.float32)
    xentropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    return xentropy_mean


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def run_gpu_train(use_pretrained_model, epoch_num):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    module_name = 'inception_v1'
    checkpoint_dir = 'checkpoint/{}_{}_{}'.format(module_name, epoch_num-1, FLAGS.alpha)

    saved_checkpoint_dir = 'checkpoint/{}_{}_{}'.format(module_name, epoch_num, FLAGS.alpha)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(saved_checkpoint_dir):
        os.makedirs(saved_checkpoint_dir)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            dataset = imagenet.get_split(
                'train', FLAGS.dataset_dir)
            dataset_val = imagenet.get_split(
                'validation', FLAGS.dataset_dir)
            global_step = slim.create_global_step()
            learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
            istraining_placeholder = tf.placeholder(tf.bool)
            network_fn = GoogLeNet.GoogLeNet(
                num_classes=(dataset.num_classes - FLAGS.labels_offset),
                weight_decay=FLAGS.weight_decay,
                is_training=istraining_placeholder)
            tower_grads = []
            logits_lst = []
            losses_lst = []
            opt = _configure_optimizer(learning_rate)
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.gpu_num, network_fn.default_image_size,
                                                                   network_fn.default_image_size, 3))
            labels_placeholder = tf.placeholder(tf.int64, shape=(FLAGS.batch_size * FLAGS.gpu_num, dataset.num_classes))
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            logits, _, _, _ = network_fn(images_placeholder[gpu_index * FLAGS.batch_size:
                                                        (gpu_index + 1) * FLAGS.batch_size])
                            logits_lst.append(logits)
                            loss = tower_loss_xentropy_dense(
                                logits,
                                labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                            )
                            losses_lst.append(loss)
                            # varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["logits"])]
                            varlist = tf.trainable_variables()
                            #print([v.name for v in varlist])
                            grads = opt.compute_gradients(loss, varlist)
                            tower_grads.append(grads)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

            image_preprocessing_fn = preprocessing_factory.get_preprocessing(is_training=True)
            val_image_preprocessing_fn = preprocessing_factory.get_preprocessing(is_training=False)

            loss_op = tf.reduce_mean(losses_lst, name='softmax')
            logits_op = tf.concat(logits_lst, 0)
            acc_op = accuracy(logits_op, labels_placeholder)
            grads = average_gradients(tower_grads)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies([tf.group(*update_ops)]):
                apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            images_op, labels_op = input_data.inputs(dataset=dataset, image_preprocessing_fn=image_preprocessing_fn, network_fn=network_fn, num_epochs=1, batch_size=FLAGS.batch_size * FLAGS.gpu_num)
            val_images_op, val_labels_op = input_data.inputs(dataset=dataset_val, image_preprocessing_fn=val_image_preprocessing_fn, network_fn=network_fn, num_epochs=None, batch_size=FLAGS.batch_size * FLAGS.gpu_num)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            print([var.name for var in bn_moving_vars])

            if use_pretrained_model:
                varlist = tf.trainable_variables()
                varlist += bn_moving_vars
                print(varlist)
                # vardict = {v.name[:-2].replace('MobileNet', 'MobilenetV1'): v for v in varlist}
                saver = tf.train.Saver(varlist)
                # saver = tf.train.Saver(vardict)
                if os.path.isfile(FLAGS.checkpoint_path):
                    saver.restore(sess, FLAGS.checkpoint_path)
                    print(
                        '#############################Session restored from pretrained model at {}!###############################'.format(
                            FLAGS.checkpoint_path))
                else:
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_path)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver = tf.train.Saver(varlist)
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print('Session restored from pretrained degradation model at {}!'.format(
                            ckpt.model_checkpoint_path))
            else:
                varlist = tf.trainable_variables()
                varlist += bn_moving_vars
                saver = tf.train.Saver(varlist)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(
                        '#############################Session restored from trained model at {}!###############################'.format(
                            ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)

            saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
            step = 0
            try:
                while not coord.should_stop():
                    start_time = time.time()
                    images, labels = sess.run([images_op, labels_op])
                    _, loss_value = sess.run([apply_gradient_op, loss_op],
                                         feed_dict={images_placeholder: images,
                                                    labels_placeholder: labels,
                                                    istraining_placeholder: True})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    duration = time.time() - start_time
                    print('Step: {:4d} time: {:.4f} loss: {:.8f}'.format(step, duration, loss_value))

                    if step % FLAGS.val_step == 0:
                        start_time = time.time()
                        images, labels = sess.run([images_op, labels_op])
                        acc, loss_value = sess.run([acc_op, loss_op],
                                               feed_dict={images_placeholder: images,
                                                          labels_placeholder: labels,
                                                          istraining_placeholder: False})
                        print("Step: {:4d} time: {:.4f}, training accuracy: {:.5f}, loss: {:.8f}".
                            format(step, time.time() - start_time, acc, loss_value))

                        start_time = time.time()
                        images, labels = sess.run([val_images_op, val_labels_op])
                        acc, loss_value = sess.run([acc_op, loss_op],
                                               feed_dict={images_placeholder: images,
                                                          labels_placeholder: labels,
                                                          istraining_placeholder: False})
                        print("Step: {:4d} time: {:.4f}, validation accuracy: {:.5f}, loss: {:.8f}".format(step, time.time() - start_time, acc, loss_value))

                    # Save a checkpoint and evaluate the model periodically.
                    if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                        checkpoint_path = os.path.join(saved_checkpoint_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training on all the examples')
            finally:
                coord.request_stop()
            coord.request_stop()
            coord.join(threads)
            checkpoint_path = os.path.join(saved_checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


    print("done")


def run_gpu_eval(use_compression=False, use_quantization=False, compute_energy=False, use_pretrained_model=True, epoch_num=0):
    from functools import reduce
    module_name = 'inception_v1'
    checkpoint_dir = 'checkpoint/{}_{}_{}'.format(module_name, epoch_num, FLAGS.alpha)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            dataset = imagenet.get_split(
                'validation', FLAGS.dataset_dir)
            istraining_placeholder = tf.placeholder(tf.bool)
            network_fn = GoogLeNet.GoogLeNet(
                num_classes=(dataset.num_classes - FLAGS.labels_offset),
                weight_decay=FLAGS.weight_decay,
                is_training=istraining_placeholder)
            logits_lst = []
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.gpu_num, network_fn.default_image_size,
                                                                   network_fn.default_image_size, 3))
            labels_placeholder = tf.placeholder(tf.int64, shape=(FLAGS.batch_size * FLAGS.gpu_num, dataset.num_classes))
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            logits, end_points, end_points_Ofmap, end_points_Ifmap = network_fn(
                                                        images_placeholder[gpu_index * FLAGS.batch_size:
                                                        (gpu_index + 1) * FLAGS.batch_size])
                            logits_lst.append(logits)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

            image_preprocessing_fn = preprocessing_factory.get_preprocessing(is_training=False)

            logits_op = tf.concat(logits_lst, 0)
            right_count_top1_op = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_op), axis=1), tf.argmax(labels_placeholder, axis=1)), tf.int32))
            right_count_topk_op = tf.reduce_sum(tf.cast(tf.nn.in_top_k(tf.nn.softmax(logits_op), tf.argmax(labels_placeholder, axis=1), 5), tf.int32))

            images_op, labels_op = input_data.inputs(dataset=dataset, image_preprocessing_fn=image_preprocessing_fn,
                                                     network_fn=network_fn, num_epochs=1, batch_size=FLAGS.batch_size*FLAGS.gpu_num)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            print([var.name for var in bn_moving_vars])

            if use_pretrained_model:
                varlist = tf.trainable_variables()
                varlist += bn_moving_vars
                print(varlist)
                saver = tf.train.Saver(varlist)
                # saver = tf.train.Saver(vardict)
                if os.path.isfile(FLAGS.checkpoint_path):
                    saver.restore(sess, FLAGS.checkpoint_path)
                    print(
                        '#############################Session restored from pretrained model at {}!###############################'.format(
                            FLAGS.checkpoint_path))
                else:
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_path)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver = tf.train.Saver(varlist)
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print('Session restored from pretrained degradation model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                varlist = tf.trainable_variables()
                varlist += bn_moving_vars
                saver = tf.train.Saver(varlist)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(
                        '#############################Session restored from trained model at {}!###############################'.format(
                            ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)

            mat_eng = matlab.engine.start_matlab()
            seed = 500
            alpha = FLAGS.alpha
            memory = 0
            for v in tf.trainable_variables() + bn_moving_vars:
                if 'weights' in v.name:
                    memory += np.prod(sess.run(v).shape)
                    print("weights.name: {}".format(v.name))
                    print("weights.shape: {}".format(sess.run(v).shape))
                    if use_compression:
                        weights = np.transpose(sess.run(v), (3, 2, 1, 0))
                        shape = weights.shape
                        n, c, w = shape[0], shape[1], shape[2]
                        k = int(alpha * n * c * w)
                        weight_clustered, mse = cluster_conv(weights, k, seed)
                        weight_clustered = np.transpose(weight_clustered, (3, 2, 1, 0))
                        sess.run(v.assign(weight_clustered))
                        print("weight_clustered shape: {}".format(weight_clustered.shape))
                        print("mse: {}".format(mse))
                        seed += 1
                    if use_quantization:
                        weights = np.transpose(sess.run(v), (3, 2, 1, 0))
                        shape = weights.shape
                        weight_quantized = mat_eng.get_fi(matlab.double(weights.tolist()),
                                                          FLAGS.bitwidth, FLAGS.bitwidth - FLAGS.bitwidth_minus_fraction_length)
                        weight_quantized = np.asarray(weight_quantized).reshape(shape).astype('float32')
                        weight_quantized = np.transpose(weight_quantized, (3, 2, 1, 0))
                        sess.run(v.assign(weight_quantized))
                        print("weight_quantized shape: {}".format(weight_quantized.shape))
                    print('=====================================')

                if any(x in v.name for x in ['beta']):
                    memory += np.prod(sess.run(v).shape)
                    print("beta.name: {}".format(v.name))
                    print("beta.shape: {}".format(sess.run(v).shape))
                    if use_quantization:
                        weights = sess.run(v)
                        shape = weights.shape
                        weight_quantized = mat_eng.get_fi(matlab.double(weights.tolist()),
                                                          FLAGS.bn_bitwidth, FLAGS.bn_bitwidth - FLAGS.bitwidth_minus_fraction_length)
                        weight_quantized = np.asarray(weight_quantized).reshape(shape).astype('float32')
                        sess.run(v.assign(weight_quantized))
                        print("beta_quantized shape: {}".format(weight_quantized.shape))
                    print('+++++++++++++++++++++++++++++++++++++')

            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=0)
            print("############################################### MEMORY IS {} ###############################################".format(memory))

            if compute_energy:
                weights_dict = {}
                for v in tf.trainable_variables():
                    if 'weights' in v.name:
                        vname = "_".join(v.name.split('/')[1:-1])
                        print("v.name: {}".format(vname))
                        print("v.shape: {}".format(sess.run(v).shape))
                        #weights = np.transpose(sess.run(v), (3, 2, 1, 0))
                        weights = sess.run(v)
                        print("v.nzeros: {}".format(np.count_nonzero(weights == 0)))
                        weights_dict[vname] = [reduce(lambda x, y: x * y, weights.shape) * (1-FLAGS.alpha), weights.shape]
                        print('=====================================')

            total_v = 0.0
            test_correct_num_top1 = 0.0
            test_correct_num_topk = 0.0

            from tqdm import tqdm

            pbar = tqdm(total=dataset.num_samples // (FLAGS.gpu_num * FLAGS.batch_size),)
            i = 1
            model_params_dict = {}
            try:
                while not coord.should_stop():
                    pbar.update(1)
                    images, labels = sess.run([images_op, labels_op])

                    right_count_top1, right_count_topk = sess.run(
                                                [right_count_top1_op, right_count_topk_op],
                                               feed_dict={images_placeholder: images,
                                                          labels_placeholder: labels,
                                                          istraining_placeholder: False})

                    end_points_Ofmap_dict, end_points_Ifmap_dict = sess.run(
                                                [end_points_Ofmap, end_points_Ifmap],
                                               feed_dict={images_placeholder: images,
                                                          labels_placeholder: labels,
                                                          istraining_placeholder: False})

                    test_correct_num_top1 += right_count_top1
                    test_correct_num_topk += right_count_topk
                    total_v += labels.shape[0]

                    if compute_energy:
                        keys = list(end_points_Ifmap_dict.keys())
                        if i == 1:
                            for k in keys:
                                model_params_dict[k] = {}
                                model_params_dict[k]["IfMap_Shape"] = end_points_Ifmap_dict[k].shape
                                model_params_dict[k]["IfMap_nZeros"] = np.count_nonzero(end_points_Ifmap_dict[k] == 0)

                                model_params_dict[k]["Filter_Shape"] = weights_dict[k][1]
                                model_params_dict[k]["Filter_nZeros"] = int(weights_dict[k][0])

                                model_params_dict[k]["OfMap_Shape"] = end_points_Ofmap_dict[k].shape
                                model_params_dict[k]["OfMap_nZeros"] = np.count_nonzero(end_points_Ofmap_dict[k] == 0)
                                print("Layer Name: {}".format(k))
                                print("IfMap Shape: {}".format(end_points_Ifmap_dict[k].shape))
                                print("IfMap nZeros: {:.4e}".format(np.count_nonzero(end_points_Ifmap_dict[k]==0)))
                                print("IfMap nZeros Avg: {:.4e}".format(model_params_dict[k]["IfMap_nZeros"]))
                                print("Filter Shape: {}".format(weights_dict[k][1]))
                                print("Filter nZeros: {:.4e}".format(int(weights_dict[k][0])))
                                print("OfMap Shape: {}".format(end_points_Ofmap_dict[k].shape))
                                print("OfMap nZeros: {:.4e}".format(np.count_nonzero(end_points_Ofmap_dict[k]==0)))
                                print("OfMap nZeros Avg: {:.4e}".format(model_params_dict[k]["OfMap_nZeros"]))
                                print('==========================================================================')
                        else:
                            for k in keys:
                                model_params_dict[k]["IfMap_nZeros"] = (model_params_dict[k]["IfMap_nZeros"]+
                                                                    np.count_nonzero(end_points_Ifmap_dict[k] == 0)/(i-1))*(i-1)/i
                                model_params_dict[k]["OfMap_nZeros"] = (model_params_dict[k]["OfMap_nZeros"]+
                                                                    np.count_nonzero(end_points_Ofmap_dict[k] == 0)/(i-1))*(i-1)/i
                        i += 1
            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()
                if compute_energy:
                    import pickle
                    with open('model_params_dict.pkl', 'wb') as f:
                        pickle.dump(model_params_dict, f, pickle.HIGHEST_PROTOCOL)
                    with open('GoogLeNet_Pruned_{}.txt'.format(FLAGS.alpha),'w') as wf:
                        for k in keys:
                            wf.write("Layer Name: {}\n".format(k))

                            wf.write("IfMap Shape: {}\n".format(model_params_dict[k]["IfMap_Shape"]))
                            wf.write("IfMap nZeros: {:.4e}\n".format(model_params_dict[k]["IfMap_nZeros"]))

                            wf.write("Filter Shape: {}\n".format(model_params_dict[k]["Filter_Shape"]))
                            wf.write("Filter nZeros: {:.4e}\n".format(model_params_dict[k]["Filter_nZeros"]))

                            wf.write("OfMap Shape: {}\n".format(model_params_dict[k]["OfMap_Shape"]))
                            wf.write("OfMap nZeros: {:.4e}\n".format(model_params_dict[k]["OfMap_nZeros"]))
                            wf.write('==========================================================================\n')
            coord.join(threads)
            print('Test acc top1:', test_correct_num_top1 / total_v,
                  'Test_correct_num top1:', test_correct_num_top1,
                  'Total_v:', total_v)
            print('Test acc topk:', test_correct_num_topk / total_v,
                  'Test_correct_num topk:', test_correct_num_topk,
                  'Total_v:', total_v)

            isCompression = lambda bool: "Compression_" if bool else "NoCompression_"
            isQuantization = lambda bool: "Quantization_" if bool else "NoQuantization_"
            with open('{}_{}_{}_{}_evaluation.txt'.format(isCompression(use_compression), isQuantization(use_quantization), epoch_num, FLAGS.alpha), 'w') as wf:
                wf.write('test acc top1:{}\ttest_correct_num top1:{}\ttotal_v:{}\n'.format(test_correct_num_top1 / total_v, test_correct_num_top1, total_v))
                wf.write('test acc topk:{}\ttest_correct_num topk:{}\ttotal_v:{}\n'.format(test_correct_num_topk / total_v, test_correct_num_topk, total_v))

    print("done")


def write_network_energy_conf():
    order_list = [
        "Conv2d_1a_7x7",
        "Conv2d_2b_1x1",
        "Conv2d_2c_3x3",
        "Mixed_3b_Branch_1_Conv2d_0a_1x1",
        "Mixed_3b_Branch_2_Conv2d_0a_1x1",
        "Mixed_3b_Branch_0_Conv2d_0a_1x1",
        "Mixed_3b_Branch_1_Conv2d_0b_3x3",
        "Mixed_3b_Branch_2_Conv2d_0b_3x3",
        "Mixed_3b_Branch_3_Conv2d_0b_1x1",
        "Mixed_3c_Branch_1_Conv2d_0a_1x1",
        "Mixed_3c_Branch_2_Conv2d_0a_1x1",
        "Mixed_3c_Branch_0_Conv2d_0a_1x1",
        "Mixed_3c_Branch_1_Conv2d_0b_3x3",
        "Mixed_3c_Branch_2_Conv2d_0b_3x3",
        "Mixed_3c_Branch_3_Conv2d_0b_1x1",
        "Mixed_4b_Branch_1_Conv2d_0a_1x1",
        "Mixed_4b_Branch_2_Conv2d_0a_1x1",
        "Mixed_4b_Branch_0_Conv2d_0a_1x1",
        "Mixed_4b_Branch_1_Conv2d_0b_3x3",
        "Mixed_4b_Branch_2_Conv2d_0b_3x3",
        "Mixed_4b_Branch_3_Conv2d_0b_1x1",
        "Mixed_4c_Branch_1_Conv2d_0a_1x1",
        "Mixed_4c_Branch_2_Conv2d_0a_1x1",
        "Mixed_4c_Branch_0_Conv2d_0a_1x1",
        "Mixed_4c_Branch_1_Conv2d_0b_3x3",
        "Mixed_4c_Branch_2_Conv2d_0b_3x3",
        "Mixed_4c_Branch_3_Conv2d_0b_1x1",
        "Mixed_4d_Branch_1_Conv2d_0a_1x1",
        "Mixed_4d_Branch_2_Conv2d_0a_1x1",
        "Mixed_4d_Branch_0_Conv2d_0a_1x1",
        "Mixed_4d_Branch_1_Conv2d_0b_3x3",
        "Mixed_4d_Branch_2_Conv2d_0b_3x3",
        "Mixed_4d_Branch_3_Conv2d_0b_1x1",
        "Mixed_4e_Branch_1_Conv2d_0a_1x1",
        "Mixed_4e_Branch_2_Conv2d_0a_1x1",
        "Mixed_4e_Branch_0_Conv2d_0a_1x1",
        "Mixed_4e_Branch_1_Conv2d_0b_3x3",
        "Mixed_4e_Branch_2_Conv2d_0b_3x3",
        "Mixed_4e_Branch_3_Conv2d_0b_1x1",
        "Mixed_4f_Branch_1_Conv2d_0a_1x1",
        "Mixed_4f_Branch_2_Conv2d_0a_1x1",
        "Mixed_4f_Branch_0_Conv2d_0a_1x1",
        "Mixed_4f_Branch_1_Conv2d_0b_3x3",
        "Mixed_4f_Branch_2_Conv2d_0b_3x3",
        "Mixed_4f_Branch_3_Conv2d_0b_1x1",
        "Mixed_5b_Branch_1_Conv2d_0a_1x1",
        "Mixed_5b_Branch_2_Conv2d_0a_1x1",
        "Mixed_5b_Branch_0_Conv2d_0a_1x1",
        "Mixed_5b_Branch_1_Conv2d_0b_3x3",
        "Mixed_5b_Branch_2_Conv2d_0a_3x3",
        "Mixed_5b_Branch_3_Conv2d_0b_1x1",
        "Mixed_5c_Branch_1_Conv2d_0a_1x1",
        "Mixed_5c_Branch_2_Conv2d_0a_1x1",
        "Mixed_5c_Branch_0_Conv2d_0a_1x1",
        "Mixed_5c_Branch_1_Conv2d_0b_3x3",
        "Mixed_5c_Branch_2_Conv2d_0b_3x3",
        "Mixed_5c_Branch_3_Conv2d_0b_1x1",
        "Logits_Conv2d_0c_1x1"]
    import pickle
    with open('model_params_dict.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        model_params_dict = u.load()
    lines = []
    i = 0
    with open('NetworkConf_GoogLeNet_Unpruned.txt','r') as f:
        for line in f:
            line = line.rstrip('\n')
            entries = line.split(',')
            entries[5] = "{:.4e}".format(model_params_dict[order_list[i]]["IfMap_nZeros"])
            entries[11] = "{:.4e}".format(model_params_dict[order_list[i]]["Filter_nZeros"])
            entries[17] = "{:.4e}".format(model_params_dict[order_list[i]]["OfMap_nZeros"])
            lines.append(','.join(entries))
            i += 1
    with open('NetworkConf_GoogLeNet_Pruned_{}.txt'.format(FLAGS.alpha),'w') as wf:
        for line in lines:
            wf.write(line+'\n')


def cluster_conv(weight, n_clusters, seed):
    from sklearn.metrics import mean_squared_error
    # weight: cuda tensor
    filters_num = weight.shape[0]
    filters_channel = weight.shape[1]
    filters_size = weight.shape[2]

    weight_vector = weight.reshape(-1, filters_size)


    weight_vector_clustered = kmeans.k_means_vector_gpu_fp32(weight_vector.astype('float32'),
                                                           n_clusters,
                                                           verbosity=0,
                                                           seed=seed,
                                                           gpu_id=0).astype('float32')
    weight_cube_clustered = weight_vector_clustered.reshape(filters_num, filters_channel,
                                                            filters_size, -1)
    mse = mean_squared_error(weight_vector, weight_vector_clustered)


    weight_compress = weight_cube_clustered.astype('float32')

    return weight_compress, mse


def get_alpha_from_compression_rate(compression_rate):
    def bin_search_k(shape_lst, cr_target):
        import math
        high = 1.0
        low = 0.001
        alpha = (high + low) / 2
        while True:
            nparams_orig = 0
            nparams_comp = 0
            for shape in shape_lst:
                n, c, w = shape[3], shape[2], shape[1]
                nparams_orig += n * c * w * w * 32
                k = int(alpha * n * c * w)
                nparams_comp += (n * c * w * int(math.log(k, 2) + 1) + k * w * 32)
            cr = nparams_orig / nparams_comp
            delta = cr_target - cr
            print('delta: {}'.format(delta))
            print('alpha: {}'.format(alpha))
            print('###################')
            if math.fabs(delta) < 0.02:
                break
            if delta < 0.0:
                low = (high + low) / 2
                alpha = (high + low) / 2
            else:
                high = (high + low) / 2
                alpha = (high + low) / 2
        return alpha

    from functools import reduce
    model_name = 'inception_v1'

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            dataset = imagenet.get_split(
                'validation', FLAGS.dataset_dir)
            istraining_placeholder = tf.placeholder(tf.bool)
            network_fn = GoogLeNet.GoogLeNet(
                model_name,
                num_classes=(dataset.num_classes - 1),
                weight_decay=FLAGS.weight_decay,
                is_training=istraining_placeholder)
            logits_lst = []
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.gpu_num, network_fn.default_image_size,
                                                                   network_fn.default_image_size, 3))
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, 1):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            logits, end_points, end_points_Ofmap, end_points_Ifmap =  (
                                                        images_placeholder[gpu_index * FLAGS.batch_size:
                                                        (gpu_index + 1) * FLAGS.batch_size])
                            logits_lst.append(logits)
                            #end_points_lst.append(end_points)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()


            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            shape_lst = []

            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    #print(sess.run(v).shape)
                    print("v.name: {}".format(v.name))
                    print("v.shape: {}".format(sess.run(v).shape))
                    print('=====================================')
                    shape_lst.append(sess.run(v).shape)

                if 'beta' in v.name:
                    print("v.name: {}".format(v.name))
                    print("v.shape: {}".format(sess.run(v).shape))
                    print('+++++++++++++++++++++++++++++++++++++')
            alpha = bin_search_k(shape_lst, compression_rate)

            print("compression rate: {:.4f}, alpha: {:.4f}".format(compression_rate, alpha))


def main(_):
    #write_network_energy_conf()
    run_gpu_eval(use_compression=True, use_quantization=False, compute_energy=False, use_pretrained_model=True, epoch_num=0)
    for epoch_num in range(1, 50):
        run_gpu_train(use_pretrained_model=False, epoch_num=epoch_num)
        run_gpu_eval(use_compression=True, use_quantization=False, compute_energy=False, use_pretrained_model=False, epoch_num=epoch_num)

if __name__ == '__main__':
  tf.app.run()
