import tensorflow as tf
import math
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tflearn.layers.conv import global_avg_pool
from IPython import embed


# Paras for BN
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'


class RECnn(object):
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, position_size, filter_sizes, num_filters, l2_reg_lambda=0.0001):
        with tf.device('/gpu:1'):
            self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
            self.input_p1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_p1")
            self.input_p2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_p2")
            self.input_ps1 = tf.placeholder(tf.int32, [None], name="input_ps1")
            self.input_ps2 = tf.placeholder(tf.int32, [None], name="input_ps2")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            with tf.name_scope("position-embedding"):
                W = tf.Variable(tf.random_uniform([62,5],
                                             minval=-math.sqrt(6/(3*position_size+3*embedding_size)),
                                             maxval=math.sqrt(6/(3*position_size+3*embedding_size))),
                                             name="W")
                self.input_x_p1 = tf.nn.embedding_lookup(W, self.input_p1)
                self.input_x_p2 = tf.nn.embedding_lookup(W, self.input_p2)
                self.input_x_ps1 = tf.nn.embedding_lookup(W, self.input_ps1)
                self.input_x_ps2 = tf.nn.embedding_lookup(W, self.input_ps2)
                self.x = tf.concat([self.input_x, self.input_x_p1, self.input_x_p2],2)
                self.embedded_chars_expanded = tf.expand_dims(self.x, -1)

            # designed for multi size of filter , but in the end I just used kernel size 3.
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, embedding_size+2*position_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    c_ = {'use_bias': True,
                          'is_training': tf.cond(self.dropout_keep_prob < 1.0, lambda: True, lambda: False)}
                    conv = self.bn(conv, c_, 'first-bn')

                    beta2 = tf.Variable(tf.truncated_normal([1], stddev=0.08), name='first-swish')
                    x2 = tf.nn.bias_add(conv, b)
                    h = x2 * tf.nn.sigmoid(x2 * beta2)

                    for i in range(4):
                        h2 = self.Cnnblock(num_filters, h, i)
                        h = h2+h

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_avg = tf.nn.avg_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
                    pooled_outputs.append(pooled_avg)

            num_filters_total = num_filters*2

            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="hidden_feature")


            with tf.name_scope("MLP"):
                W0 = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W0")
                b0 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b0")
                h0 = tf.nn.relu(tf.nn.xw_plus_b(self.h_pool_flat, W0, b0))
                W1 = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b1")
                self.h1 = tf.nn.relu(tf.nn.xw_plus_b(h0, W1, b1))

            # Add dropout
            with tf.name_scope("dropout"):
                self.h1 = tf.nn.dropout(self.h1, self.dropout_keep_prob)

            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                self.scores = tf.nn.xw_plus_b(self.h1, W, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)

                self.l2_loss_in = tf.contrib.layers.apply_regularization(
                    regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda),
                    weights_list=tf.trainable_variables())

                self.loss = tf.reduce_mean(losses) + self.l2_loss_in
                tf.summary.scalar('final_loss', self.loss)

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def Cnnblock(self, num_filters, h, i, has_se=True):
        W1 = tf.get_variable(
            "W1_"+str(i),
            shape=[3, 1, num_filters, num_filters],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1_"+str(i))
        conv1 = tf.nn.conv2d(
            h,
            W1,
            strides=[1,1,1,1],
            padding="SAME")

        c_ = {'use_bias':True, 'is_training':tf.cond(self.dropout_keep_prob<1.0, lambda :True, lambda :False)}
        conv1 = self.bn(conv1, c_, str(i) + '-conv1')

        beta1 = tf.Variable(tf.truncated_normal([1], stddev=0.08), name='swish-beta-{}-1'.format(i))
        x1 = tf.nn.bias_add(conv1, b1)
        h1 = x1 * tf.nn.sigmoid(x1 * beta1)

        W2 = tf.get_variable(
            "W2_"+str(i),
            shape=[3, 1, num_filters, num_filters],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2_"+str(i))
        conv2 = tf.nn.conv2d(
            h1,
            W2,
            strides=[1,1,1,1],
            padding="SAME")

        conv2 = self.bn(conv2, c_, str(i) + '-conv2')

        beta2 = tf.Variable(tf.truncated_normal([1], stddev=0.08), name='swish-beta-{}-2'.format(i))
        x2 = tf.nn.bias_add(conv2, b2)
        h2 = x2 * tf.nn.sigmoid(x2 * beta2)

        if has_se:
            h2 = self.Squeeze_excitation_layer(h2, num_filters, 16, 'se-block-' + str(i))

        return h2


    def bn(self, x, c, name):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        if c['use_bias']:
            bias = self._get_variable('bn_bias_{}'.format(name), params_shape,
                                 initializer=tf.zeros_initializer)
            return x + bias

        axis = list(range(len(x_shape) - 1))
        beta = self._get_variable('bn_beta_{}'.format(name),
                             params_shape,
                             initializer=tf.zeros_initializer)
        gamma = self._get_variable('bn_gamma_{}'.format(name),
                              params_shape,
                              initializer=tf.ones_initializer)
        moving_mean = self._get_variable('bn_moving_mean_{}'.format(name),
                                    params_shape,
                                    initializer=tf.zeros_initializer,
                                    trainable=False)
        moving_variance = self._get_variable('bn_moving_variance_{}'.format(name),
                                        params_shape,
                                        initializer=tf.ones_initializer,
                                        trainable=False)
        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            c['is_training'], lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

        return x

    def _get_variable(self, name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype='float',
                      trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               collections=collections,
                               trainable=trainable)
    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = self.Global_Average_Pooling(input_x)

            excitation = self.Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
            excitation = self.Relu(excitation)
            excitation = self.Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
            excitation = self.Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

            return scale

    def Global_Average_Pooling(self, x):
        return global_avg_pool(x, name='Global_avg_pooling')

    def Relu(self, x):
        return tf.nn.relu(x)

    def Sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def Fully_connected(self, x, units, layer_name='fully_connected'):
        with tf.name_scope(layer_name):
            return tf.layers.dense(inputs=x, use_bias=True, units=units)

