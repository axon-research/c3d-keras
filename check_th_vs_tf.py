#!/usr/bin/env python

from keras import backend as K
import c3d_model
import numpy as np
import sys

def reindex(x):
    # https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py#L90-L115
    # invert the last three axes
    if x.ndim != 5:
        print "[Error] Input to reindex must be 5D nparray."
        return None

    N = x.shape[0]
    C = x.shape[1]
    L = x.shape[2]
    H = x.shape[3]
    W = x.shape[4]
    y = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for l in range(L):
                for h in range(H):
                    for w in range(W):
                        y[n, c, l, h, w] = x[n, c,
                                                   L - l - 1,
                                                   H - h - 1,
                                                   W - w - 1]
    return y

def main():
    # build model in TH mode, as th_model
    K.set_image_dim_ordering('th')
    th_model = c3d_model.get_model(backend='th')
    th_model.load_weights('./models/sports1M_weights_th.h5')

    # build model in TF mode, as tf_model
    K.set_image_dim_ordering('tf')
    tf_model = c3d_model.get_model(backend='tf')
    tf_model.load_weights('./models/sports1M_weights_tf.h5')

    # check weights for th_model vs tf_model
    for th_layer, tf_layer in zip(th_model.layers, tf_model.layers):
       if th_layer.__class__.__name__ == 'Convolution3D':
           th_kernel, th_bias = th_layer.get_weights()
           if tf_layer.__class__.__name__ != 'Convolution3D':
               print "[Panic] layer mismatch!"
               sys.exit(-1)
           tf_kernel, tf_bias = tf_layer.get_weights()
           print "[Info] tf_kernel shape={}, th_kernel shape={}".format(
                   tf_kernel.shape,
                   th_kernel.shape)
           th_to_tf = reindex(th_kernel)
           th_to_tf = np.transpose(th_to_tf, (2, 3, 4, 1, 0))
           delta = tf_kernel - th_to_tf
           print "[Info] delta max,mean,min={},{},{}".format(
                   np.max(delta),
                   np.mean(delta),
                   np.min(delta),
                   )

if __name__ == '__main__':
    main()
