# Copyright 2017 Chenxi Liu. All Rights Reserved.
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

# sample usage:
# python resnet_main.py 0 single

import tensorflow as tf
import numpy as np
import resnet_model
from PIL import Image
import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import pdb

def process_im(imname, mu):
  im = np.array(Image.open(imname), dtype=np.float32)
  if im.ndim == 3:
    if im.shape[2] == 4:
      im = im[:, :, 0:3]
    im = im[:,:,::-1]
  else:
    im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
  im -= mu
  im = np.expand_dims(im, axis=0)
  return im

if __name__ == "__main__":

  caffe_root = '/media/Work_HD/cxliu/tools/caffe/'
  mu = np.array((104.00698793, 116.66876762, 122.67891434))
  pretrained_model = './model/ResNet101_init.tfmodel'

  atrous = False
  if sys.argv[2] == 'atrous':
    atrous = True
  model = resnet_model.ResNet(atrous=atrous)

  snapshot_restorer = tf.train.Saver()
  sess = tf.Session()
  snapshot_restorer.restore(sess, pretrained_model)

  if sys.argv[2] == 'single':
    im = process_im('example/cat.jpg', mu)
    pred = sess.run(model.pred, feed_dict={
              model.images  : im,
              model.labels  : np.zeros((1, 1000)) # dummy
          })
    pred = pred.squeeze()
    labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
    labels_name = np.loadtxt(labels_file, str, delimiter='\t')
    print 'output label:', labels_name[pred.argmax()]

  elif sys.argv[2] == 'atrous':
    im = process_im('example/cat.jpg', mu)
    pred = sess.run(model.pred, feed_dict={
              model.images  : im,
              model.labels  : np.zeros((1, 1000)) # dummy
          })
    print 'output size:', np.shape(pred)

  elif sys.argv[2] == 'imagenet':
    imagenet_val_dir = '/media/Work_HD/cxliu/datasets/imagenet/ILSVRC2012_img_val/'
    imagenet_val_gt = '/media/Work_HD/cxliu/tools/caffe/data/ilsvrc12/val.txt'
    labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
    labels_name = np.loadtxt(labels_file, str, delimiter='\t')
    lines = np.loadtxt(imagenet_val_gt, str, delimiter='\n')
    c1, c5 = 0, 0
    for i, line in enumerate(lines):
      [imname, label] = line.split(' ')
      label = int(label)
      im = process_im(imagenet_val_dir + imname, mu)
      pred = sess.run(model.pred, feed_dict={
          model.images  : im,
          model.labels  : np.zeros((1, 1000)) # dummy
      })
      pred = pred.squeeze()
      top5 = pred.argsort()[::-1][0:5]
      top1 = top5[0]
      if label == top1:
        c1 += 1
      if label in top5:
        c5 += 1
      print('images: %d\ttop 1: %0.4f\ttop 5: %0.4f' % (i + 1, c1/(i + 1.), c5/(i + 1.)))