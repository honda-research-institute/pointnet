import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from progress.bar import Bar # sudo pip3 install progress

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import models
from models.pointnet_basic_ndt import PNetBasicNDT
import utils
import utils.tf_util
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--dataset_dir', default='/home/jhuang/Kitti/jhuang',
                    help='Dataset dir [default: /home/jhuang/Kitti/jhuang]')
parser.add_argument('--vox_size', type=float, default=0.5, help='Voxel size in m [default: 0.5]')
parser.add_argument('--num_point', type=int, default=32, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
DATASET_DIR = FLAGS.dataset_dir
VOX_SIZE = FLAGS.vox_size
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

NUM_CLASSES = 2
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# Get train/test split
TRAIN_FILES = provider.getDataFiles(os.path.join(DATASET_DIR, 'train.lst'))
TEST_FILES = provider.getDataFiles(os.path.join(DATASET_DIR, 'test.lst'))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

# TRAIN_FILES = [[filename_0, label_0], [filename_1, label_1], ...]
def get_all_train_data():
    all_data, all_labels = np.empty((0, NUM_POINT, 7)), np.empty((0))
    bar = Bar('Parsing train files', max=20, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
    bar.max = len(TRAIN_FILES)
    for fn in range(len(TRAIN_FILES)):
        #print("{}/{}".format(fn, len(TRAIN_FILES)))
        bar.next()
        fname = os.path.join(DATASET_DIR, TRAIN_FILES[fn][0])
        label = TRAIN_FILES[fn][1]
        current_data, current_labels = provider.loadDataFile((fname, label), VOX_SIZE, NUM_POINT)
        current_data = current_data[np.newaxis, :]  # make size=BxNx7
        all_data = np.concatenate((all_data, current_data))
        all_labels = np.concatenate((all_labels, current_labels))
    bar.finish()
    return all_data, all_labels

# TRAIN_FILES = [[filename_0, label_0], [filename_1, label_1], ...]
def get_all_test_data():
    all_data, all_labels = np.empty((0, NUM_POINT, 7)), np.empty((0))
    bar = Bar('Parsing test files', max=20, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
    bar.max = len(TEST_FILES)
    for fn in range(len(TEST_FILES)):
        #print("{}/{}".format(fn, len(TEST_FILES)))
        bar.next()
        fname = os.path.join(DATASET_DIR, TRAIN_FILES[fn][0])
        label = TRAIN_FILES[fn][1]
        current_data, current_labels = provider.loadDataFile((fname, label), VOX_SIZE, NUM_POINT)
        current_data = current_data[np.newaxis, :]  # make size=BxNx7
        all_data = np.concatenate((all_data, current_data))
        all_labels = np.concatenate((all_labels, current_labels))
    bar.finish()
    return all_data, all_labels

def train():
    # TRAIN_FILES = [[filename_0, label_0], [filename_1, label_1], ...]
    all_train_data, all_train_labels = get_all_train_data()
    all_test_data, all_test_labels = get_all_test_data()

    pnet_ndt = PNetBasicNDT(FLAGS.batch_size, FLAGS.num_point)
    with pnet_ndt.get_graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, is_training_pl = pnet_ndt.get_input_pls()
            train_op, pred, loss = pnet_ndt.get_output_tensors()

            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch_pl = tf.Variable(0)
            pnet_ndt.batch = batch_pl
            bn_decay = get_bn_decay(batch_pl)
            pnet_ndt.bn_decay = bn_decay
            tf.summary.scalar('bn_decay', bn_decay)

            # Get training operator
            learning_rate = get_learning_rate(batch_pl)
            tf.summary.scalar('learning_rate', learning_rate)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        sess.run(pnet_ndt.init_op, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch_pl}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, all_train_data, all_train_labels, train_writer)
            eval_one_epoch(sess, ops, all_test_data, all_test_labels, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, all_data, all_labels, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    n_samples = len(TRAIN_FILES)
    num_batches = n_samples // BATCH_SIZE

    train_file_idxs = np.arange(0, n_samples)
    np.random.shuffle(train_file_idxs)
    all_data = all_data[train_file_idxs, ...]  # BxNUM_POINTx7
    all_labels = all_labels[train_file_idxs, ...]  # B

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        # Augment batched point clouds by rotation and jittering
        # rotated_data = provider.rotate_point_cloud()
        # jittered_data = provider.jitter_point_cloud(rotated_data)

        current_data = all_data[start_idx:end_idx, :, :]
        current_labels = all_labels[start_idx:end_idx]
        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_labels,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_labels[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, all_data, all_labels, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    n_samples = len(TRAIN_FILES)
    num_batches = n_samples // BATCH_SIZE
    train_file_idxs = np.arange(0, n_samples)
    all_data = all_data[train_file_idxs, ...]  # BxNUM_POINTx7
    all_labels = all_labels[train_file_idxs, ...]  # B

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        current_data = all_data[start_idx:end_idx, :, :]
        current_labels = all_labels[start_idx:end_idx]
        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_labels,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_labels[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val * BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = current_labels[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i - start_idx] == l)


    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string(
    'eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
