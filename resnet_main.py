import tensorflow as tf
import numpy as np
import resnet_model
from PIL import Image
import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import matplotlib.pyplot as plt
import pdb

# sample usage:
# python resnet_main.py 0 single

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
  mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
  mu = mu.mean(1).mean(1)
  pretrained_model = './model/ResNet101_init.tfmodel'

  hps = resnet_model.HParams(batch_size = 1,
                num_classes = 1000,
                min_lrn_rate = 0.0001,
                lrn_rate = 0.1,
                num_residual_units = [3, 4, 23, 3],
                use_bottleneck = True,
                weight_decay_rate = 0.0002,
                relu_leakiness = 0.0,
                filters = [64, 256, 512, 1024, 2048],
                optimizer = 'mom')
  model = resnet_model.ResNet(hps, 'eval')

  snapshot_restorer = tf.train.Saver()
  sess = tf.Session()
  snapshot_restorer.restore(sess, pretrained_model)

  if sys.argv[2] == 'single':
    im = process_im('example/cat.jpg', mu)
    pred = sess.run(model.predictions, feed_dict={
              model.images  : im,
              model.labels  : np.zeros((1, 1000)) # dummy
          })
    labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
    labels_name = np.loadtxt(labels_file, str, delimiter='\t')
    print 'output label:', labels_name[pred.argmax()]

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
      pred = sess.run(model.predictions, feed_dict={
          model.images  : im,
          model.labels  : np.zeros((1, 1000)) # dummy
      })
      top1 = pred.argmax()
      top5 = pred[0].argsort()[::-1][0:5]
      if label == top1:
        c1 += 1
      if label in top5:
        c5 += 1
      print('images: %d\ttop 1: %0.4f\ttop 5 %0.4f' % (i + 1, c1/(i + 1.), c5/(i + 1.)))