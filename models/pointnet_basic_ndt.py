import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

class PNetBasicNDT(object):
    def __init__(self, batch_size, num_point, learning_rate=0.001, device=0):
        self.batch_size = batch_size
        self.num_point = num_point
        self.learning_rate = learning_rate
        self.g = tf.Graph()

        ## build the model
        with self.g.as_default():
            with tf.device('/gpu:'+str(device)):
                self.init_op = tf.global_variables_initializer()
                self._set_up_input_pls()
                self.build()
                self.define_loss()


    def _set_up_input_pls(self):
        self.pointclouds_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_point, 6))
        self.labels_pl = tf.placeholder(tf.int32, shape=(self.batch_size))
        self.batch = tf.Variable(0)  # incremented after each batch
        self.is_training_pl = tf.placeholder(tf.bool, shape=())

    def build(self):
        """ Classification PointNet, input is BxNx3, output Bx2 """
        self.end_points = {}

        # TF expects input: 4-D tensor variable BxHxWxC
        input_image = tf.expand_dims(self.pointclouds_pl, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1, 6],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl,
                             scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl,
                             scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [self.num_point, 1],
                                 padding='VALID', scope='maxpool')

        # MLP on global point cloud vector
        net = tf.reshape(net, [self.batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=self.is_training_pl,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=self.is_training_pl,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=self.is_training_pl,
                              scope='dp1')
        self.net = tf_util.fully_connected(net, 2, activation_fn=None, scope='fc3')

    def define_loss(self):
        """ pred: B*NUM_CLASSES,
            label: B, """
        pred = self.net
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_pl)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.batch)
        tf.summary.scalar('loss', self.loss)

        self.classify_loss = tf.reduce_mean(self.loss) # mean loss of this batch
        tf.summary.scalar('classify loss', self.classify_loss)

        correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(self.labels_pl))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(self.batch_size)
        tf.summary.scalar('accuracy', accuracy)

    def get_input_pls(self):
        return self.pointclouds_pl, self.labels_pl, self.batch, self.is_training_pl

    def get_output_tensors(self):
        return self.train_op, self.net, self.loss