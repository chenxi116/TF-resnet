import numpy as np
import tensorflow as tf
import resnet_model
import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import pdb

# sample usage:
# python npy2tfmodel.py 0 ./model/ResNet101.npy ./model/ResNet101_init.tfmodel

weights = np.load(sys.argv[2])[()]
# fully connected -> fully convolutional
weights['fc1000/DW'] = np.expand_dims(
  np.expand_dims(weights['fc1000/DW'], axis=0), axis=0)

model = resnet_model.ResNet(atrous=False)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
var_list = tf.all_variables()
count = 0
for item in var_list:
    item_name = item.name[7:-2] # "ResNet/" at beginning, ":0" at last
    if not item_name in weights.keys():
        continue
    print item_name
    count += 1
    sess.run(tf.assign(item, weights[item_name]))
assert(count == len(weights))

snapshot_saver = tf.train.Saver()
snapshot_saver.save(sess, sys.argv[3])
# pdb.set_trace()