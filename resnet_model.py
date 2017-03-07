# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple
from tensorflow.python.training import moving_averages
import numpy as np
import tensorflow as tf
import pdb


class ResNet(object):
  """ResNet model."""

  def __init__(self, batch_size = 1,
                     num_classes = 1000,
                     lrn_rate = 0.1,
                     num_residual_units = [3, 4, 23, 3],
                     use_bottleneck = True,
                     weight_decay_rate = 0.0001,
                     bn = False,
                     relu_leakiness = 0.0,
                     filters = [64, 256, 512, 1024, 2048],
                     atrous = False,
                     optimizer = 'mom',
                     images = tf.placeholder(tf.float32),
                     labels = tf.placeholder(tf.float32),
                     mode = 'eval'):
    """ResNet constructor.

    Args:
      : Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.lrn_rate = lrn_rate
    self.num_residual_units = num_residual_units
    self.use_bottleneck = use_bottleneck
    self.weight_decay_rate = weight_decay_rate
    self.relu_leakiness = relu_leakiness
    self.filters = filters
    self.atrous = atrous
    self.bn = bn
    self.optimizer = optimizer
    self.mode = mode
    with tf.variable_scope("ResNet"):
      self.build_graph()

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    # self.summaries = tf.summary.merge_all()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('group_1'):
      x = self.images
      x = self._conv('conv1', x, 7, 3, 64, self._stride_arr(2))
      x = self._batch_norm('bn_conv1', x)
      x = self._relu(x, self.relu_leakiness)
      x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    res_func = self._bottleneck_residual
    filters = self.filters

    with tf.variable_scope('group_2_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(1))
    for i in xrange(1, self.num_residual_units[0]):
      with tf.variable_scope('group_2_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1))

    with tf.variable_scope('group_3_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(2))
    for i in xrange(1, self.num_residual_units[1]):
      with tf.variable_scope('group_3_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1))

    if self.atrous == False:
      with tf.variable_scope('group_4_0'):
        x = res_func(x, filters[2], filters[3], self._stride_arr(2))
      for i in xrange(1, self.num_residual_units[2]):
        with tf.variable_scope('group_4_%d' % i):
          x = res_func(x, filters[3], filters[3], self._stride_arr(1))

      with tf.variable_scope('group_5_0'):
        x = res_func(x, filters[3], filters[4], self._stride_arr(2))
      for i in xrange(1, self.num_residual_units[3]):
        with tf.variable_scope('group_5_%d' % i):
          x = res_func(x, filters[4], filters[4], self._stride_arr(1))

    else:
      with tf.variable_scope('group_4_0'):
        x = res_func(x, filters[2], filters[3], self._stride_arr(1), 2)
      for i in xrange(1, self.num_residual_units[2]):
        with tf.variable_scope('group_4_%d' % i):
          x = res_func(x, filters[3], filters[3], self._stride_arr(1), 2)

      with tf.variable_scope('group_5_0'):
        x = res_func(x, filters[3], filters[4], self._stride_arr(1), 4)
      for i in xrange(1, self.num_residual_units[3]):
        with tf.variable_scope('group_5_%d' % i):
          x = res_func(x, filters[4], filters[4], self._stride_arr(1), 4)

    with tf.variable_scope('group_last'):
      x = self._relu(x, self.relu_leakiness)
      if self.atrous == False:
        x = self._global_avg_pool(x)
      self.res5c = x

    with tf.variable_scope('fc1000'):
      self.logits = self._fully_convolutional(x, self.num_classes)
      logits_flat = tf.reshape(self.logits, [-1, self.num_classes])
      pred = tf.nn.softmax(logits_flat)
      self.pred = tf.reshape(pred, tf.shape(self.logits))

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
    self.cost = tf.reduce_mean(xent, name='xent')
    self.cost += self._decay()
    # tf.summary.scalar('cost', self.cost)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.learning_rate = tf.constant(self.lrn_rate, tf.float32)
    # tf.summary.scalar('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    elif self.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
    else:
      raise NameError("Unknown optimizer type %s!" % self.optimizer)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_step = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32),
          trainable=False)
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32),
          trainable=False)

      if self.bn:
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        # tf.summary.histogram(mean.op.name, mean)
        # tf.summary.histogram(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _bottleneck_residual(self, x, in_filter, out_filter, stride, atrous=1):
    """Bottleneck residual unit with 3 sub layers."""

    orig_x = x

    with tf.variable_scope('block_1'):
      x = self._conv('conv', x, 1, in_filter, out_filter/4, stride, atrous)
      x = self._batch_norm('bn', x)
      x = self._relu(x, self.relu_leakiness)

    with tf.variable_scope('block_2'):
      x = self._conv('conv', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1], atrous)
      x = self._batch_norm('bn', x)
      x = self._relu(x, self.relu_leakiness)

    with tf.variable_scope('block_3'):
      x = self._conv('conv', x, 1, out_filter/4, out_filter, [1, 1, 1, 1], atrous)
      x = self._batch_norm('bn', x)

    with tf.variable_scope('block_add'):
      if in_filter != out_filter:
        orig_x = self._conv('conv', orig_x, 1, in_filter, out_filter, stride, atrous)
        orig_x = self._batch_norm('bn', orig_x)
      x += orig_x
      x = self._relu(x, self.relu_leakiness)

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides, atrous=1):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      if atrous == 1:
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')
      else:
        assert(strides == self._stride_arr(1))
        return tf.nn.atrous_conv2d(x, kernel, rate=atrous, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.batch_size, -1])
    w = tf.get_variable(
        'DW', [self.filters[-1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _fully_convolutional(self, x, out_dim):
    """FullyConvolutional layer for final output."""
    w = tf.get_variable(
        'DW', [1, 1, self.filters[-1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())  
    return tf.nn.conv2d(x, w, self._stride_arr(1), padding='SAME') + b 

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.expand_dims(tf.expand_dims(tf.reduce_mean(x, [1, 2]), 0), 0)
