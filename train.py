# -*- coding:utf-8 -*

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from Cnn_new import RECnn
from test import test
from util.DataManager import DataManager
from IPython import embed

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_integer("sequence_length", 100, "Sequence length (default: 100)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ====================
datamanager = DataManager(FLAGS.sequence_length)
training_data = datamanager.load_training_data()
training_data = np.array(training_data)
testing_data = datamanager.load_testing_data()
print("train set and test set size are:")
print(str(len(training_data))+" "+str(len(testing_data)))

# Random shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(training_data)))
training_data = training_data[shuffle_indices]
print("Finish randomize data")

train = training_data
dev = training_data[-2:]


# Start Training
# ====================
print("Start Training")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = RECnn(
            FLAGS.sequence_length,
            len(datamanager.relations),
            FLAGS.embedding_dim,
            5,
            list(map(int, FLAGS.filter_sizes.split(","))),
            FLAGS.num_filters,
            FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('train_loss', sess.graph)

        print("Initialize variables.")
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # I was going to use ps1, ps2 as piece wise pooling separator, but I didn't use those in the end,
        # so they are useless.
        def train_step(x_batch, y_batch, p1_batch, p2_batch, ps1, ps2):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_p1: p1_batch,
                cnn.input_p2: p2_batch,
                cnn.input_ps1:ps1,
                cnn.input_ps2:ps2,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
            _, step, loss, accuracy, summary = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy, merged_summary],
                feed_dict)
            summary_writer.add_summary(summary, step)
            print("train step:")
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return loss

        def dev_step(x_batch, y_batch, p1_batch, p2_batch, ps1, ps2):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_p1: p1_batch,
                cnn.input_p2: p2_batch,
                cnn.input_ps1: ps1,
                cnn.input_ps2: ps2,
                cnn.dropout_keep_prob: 1,
            }
            _, step, loss, accuracy, summary= sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy, merged_summary],
                feed_dict)
            summary_writer.add_summary(summary, step)
            print("dev step:")
            print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))


        # Generate batches
        batches = datamanager.batch_iter(
            train, FLAGS.batch_size, FLAGS.num_epochs)
        num_batches_per_epoch = int(len(train)/FLAGS.batch_size) + 1
        print("Batch data")
        # Training loop. For each batch...
        num_batch = 1
        num_epoch = 1
        dev_x_batch = datamanager.generate_x(dev)
        dev_p1_batch, dev_p2_batch, dev_ps1, dev_ps2 = datamanager.generate_p(dev)
        dev_y_batch = datamanager.generate_y(dev)
        for batch in batches:
            num_batch += 1
            x_batch = datamanager.generate_x(batch)
            p1_batch, p2_batch, ps1, ps2 = datamanager.generate_p(batch)
            y_batch = datamanager.generate_y(batch)
            loss = train_step(x_batch, y_batch, p1_batch, p2_batch, ps1, ps2)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("Num_batch: {}".format(num_batch))
                print("Num_epoch: {}".format(num_epoch))
                dev_step(dev_x_batch, dev_y_batch, dev_p1_batch, dev_p2_batch, dev_ps1, dev_ps2)

            # when finished one epoch, do a test
            if num_batch == num_batches_per_epoch:
                num_epoch += 1
                num_batch = 1
                test(testing_data, cnn.input_x, cnn.input_p1, cnn.input_p2, cnn.input_ps1, cnn.input_ps2, cnn.scores, cnn.predictions, cnn.dropout_keep_prob, datamanager, sess, num_epoch)
