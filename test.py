import tensorflow as tf
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from models.pointnet_basic_ndt import PNetBasicNDT
import provider

LOG_DIR = "log"
GPU_INDEX = 0
BATCH_SIZE = 1
NUM_POINT = 32
VOX_SIZE = 0.5
MODEL = PNetBasicNDT()


def test(bin_file):
    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver = tf.train.import_meta_graph(os.path.join(LOG_DIR, "model.ckpt.meta"))
    saver.restore(sess, os.path.join(LOG_DIR, "model.ckpt"))
    print("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    test_one_sample(sess, ops, bin_file)


def test_one_sample(sess, ops, bin_file):
    is_training = False
    label = 1
    # Get ndt(Nx7) and label(int)
    ndt, gt_label = provider.loadDataFile((bin_file, label), VOX_SIZE, NUM_POINT)
    ndt = ndt[np.newaxis, :]  # make size=1xNUM_POINTx7
    feed_dict = {ops['pointclouds_pl']: ndt,
                 ops['labels_pl']: gt_label,
                 ops['is_training_pl']: is_training}
    pred_val = sess.run([ops['pred']], feed_dict=feed_dict)
    print("pred: {:.2f}, gt_label: 1".format(pred_val[0, 1]))

if __name__ == "__main__":
    test(sys.argv[1])
