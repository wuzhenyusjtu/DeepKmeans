
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import inception_utils
from nets import inception_v1

slim = tf.contrib.slim

def GoogLeNet(num_classes, weight_decay=0.0, is_training=False):
  func = inception_v1.inception_v1
  arg_scope = inception_utils.inception_arg_scope(weight_decay=weight_decay)
  def network_fn(images, **kwargs):
    with slim.arg_scope(arg_scope):
      return inception_v1.inception_v1(images, num_classes, is_training=is_training)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size
  return network_fn