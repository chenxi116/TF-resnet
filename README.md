# TF-resnet

This is a Tensorflow implementation of ResNet. 

Currently it only supports testing the 101 layer model by converting the caffemodel provided by Kaiming. Although supporting other ResNet variants and training should be quick and easy. 

The `caffemodel2npy.py` is modified from [here](https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/loadcaffe.py), and the `resnet_model.py` is modified from [here](https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py).

## Example Usage
- Download the prototxt and caffemodel [provided by Kaiming](https://github.com/KaimingHe/deep-residual-networks)
- Convert caffemodel to npy file
```bash
python caffemodel2npy.py ../ResNet/ResNet-101-deploy.prototxt ../ResNet/ResNet-101-model.caffemodel ./model/ResNet101.npy
```
- Convert npy file to tfmodel
```bash
python npy2tfmodel.py 0 ./model/ResNet101.npy ./model/ResNet101_init.tfmodel
```
- Test on a single image
```bash
python resnet_main.py 0 single
```

## Performance

The converted ResNet 101 model achieves top 5 error of 7.48% and top 1 error of 24.58% on ILSVRC12 validation set. This is without any cropping/flipping/multi-scale, using only the original image.

## TODO

- Support ResNet 50 and 152
- Training code