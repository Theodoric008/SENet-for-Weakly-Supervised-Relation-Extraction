# -*- coding:utf-8 -*

import tensorflow as tf
import numpy as np
import os
from Cnn_new import RECnn
from test import test
from util.DataManager import DataManager
from IPython import embed
from sentence_aug import random_word_dropout, random_sentence_flip, random_lin_adv_noise
import kenlm
from nltk.tag import StanfordNERTagger
import pickle as pkl


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)

os.system("mkdir -p temp")

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_integer("sequence_length", 100, "Sequence length (default: 100)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0001)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 300)")

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

# load pre-augmented data from a pkl file
pkl_file = "noNA_WN100_aug_data_dict_entity_pair2aug_sentences.pkl"
pkl_dict = pkl.load(open(pkl_file, "rb"))


def data_aug(batch_, random_word_dropout_prob=0.1,
             random_sentence_flip_prob=0.0,
             random_lin_adv_prob=0.5):

    def aug(sentence):
        if random_lin_adv_prob > 0:
            sentence.words = random_lin_adv_noise(sentence.words, sentence.entity1, sentence.entity2,
                                                  pkl_dict, random_lin_adv_prob)
        if random_word_dropout_prob > 0:
            sentence.words = random_word_dropout(sentence.entity1, sentence.entity2,
                                                 sentence.words, random_word_dropout_prob)
        if random_sentence_flip_prob > 0:
            sentence.words = random_sentence_flip(sentence.words, random_sentence_flip_prob)
        return sentence.words

    res = []
    for data_item in batch_:
        if data_item.relation.id == 0:
            res.append(data_item)
        else:
            data_item.words = aug(data_item)
            res.append(data_item)
    return res


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

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        # timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        # print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)
        # saver = tf.train.Saver(tf.global_variables())

        # merged_summary = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter('train_loss', sess.graph)

        print("Initialize variables.")
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch, p1_batch, p2_batch, num_epoch, num_batches_per_epoch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_p1: p1_batch,
                cnn.input_p2: p2_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            epoch_step = step % num_batches_per_epoch
            if step % 1 == 0:
                print("num_epoch {} epoch_step {} / {} loss {:g}, acc {:g}"
                      .format(num_epoch, epoch_step, num_batches_per_epoch, loss, accuracy))
            return loss

        batches = datamanager.batch_iter(
            train, FLAGS.batch_size, FLAGS.num_epochs)
        num_batches_per_epoch = int(len(train)/FLAGS.batch_size) + 1
        print("Batch data")
        num_batch = 1
        num_epoch = 1
        for batch in batches:
            batch = data_aug(batch)
            num_batch += 1
            x_batch = datamanager.generate_x(batch)
            p1_batch, p2_batch = datamanager.generate_p(batch)
            y_batch = datamanager.generate_y(batch)
            loss = train_step(x_batch, y_batch, p1_batch, p2_batch, num_epoch=num_epoch,
                              num_batches_per_epoch=num_batches_per_epoch)
            current_step = tf.train.global_step(sess, global_step)

            # when finished one epoch, do a test
            if num_batch == num_batches_per_epoch:
                num_epoch += 1
                num_batch = 1
                if num_epoch % 1 == 0 and (num_epoch > 20 or num_epoch == 5):
                    test(testing_data, cnn.input_x, cnn.input_p1, cnn.input_p2, cnn.scores, cnn.predictions,
                         cnn.dropout_keep_prob, datamanager, sess, num_epoch)

