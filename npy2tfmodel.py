import numpy as np
import tensorflow as tf
import resnet_model
import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import pdb

# sample usage:
# python npy2tfmodel.py 0 ./model/ResNet101.npy ./model/ResNet101_init.tfmodel

weights = np.load(sys.argv[2])[()]

hps = resnet_model.HParams(batch_size = 1,
              num_classes = 1000,
              min_lrn_rate = 0.0001,
              lrn_rate = 0.1,
              num_residual_units = [3, 4, 23, 3],
              use_bottleneck = True,
              weight_decay_rate = 0.0002,
              relu_leakiness = 0.0,
              filters = [64, 256, 512, 1024, 2048],
              atrous = False,
              optimizer = 'mom')
model = resnet_model.ResNet(hps, 'eval')

sess = tf.Session()
sess.run(tf.initialize_all_variables())
var_list = tf.all_variables()
count = 0
for item in var_list:
    if not item.name[0:-2] in weights.keys():
        continue
    print item.name[0:-2]
    count += 1
    sess.run(tf.assign(item, weights[item.name[0:-2]]))
assert(count == len(weights))

snapshot_saver = tf.train.Saver()
snapshot_saver.save(sess, sys.argv[3])
# pdb.set_trace()