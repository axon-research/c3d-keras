#!/usr/bin/env python

from keras import backend as K
import c3d_model
import numpy as np

K.set_image_dim_ordering('th')

# build model in TH mode, as th_model
th_model = c3d_model.get_model(backend='th')
# load weights that were saved in TH mode into th_model
th_model.load_weights('./models/sports1M_weights_albertomontesg.h5')

K.set_image_dim_ordering('tf')

# build model in TF mode, as tf_model
tf_model = c3d_model.get_model(backend='tf')

# transfer weights from th_model to tf_model
for th_layer, tf_layer in zip(th_model.layers, tf_model.layers):
   if th_layer.__class__.__name__ == 'Convolution2D':
      kernel, bias = th_layer.get_weights()
      kernel = np.transpose(kernel, (2, 3, 1, 0))
      tf_layer.set_weights([kernel, bias])
   elif th_layer.__class__.__name__ == 'Convolution3D':
       kernel, bias = th_layer.get_weights()
       #kernel = np.transpose(kernel, (4, 2, 3, 1, 0))
       #kernel = np.transpose(kernel, (2, 4, 3, 1, 0))
       #kernel = np.transpose(kernel, (4, 3, 2, 1, 0))
       kernel = np.transpose(kernel, (2, 3, 4, 1, 0))
       tf_layer.set_weights([kernel, bias])
   else:
       tf_layer.set_weights(tf_layer.get_weights())

tf_model.save_weights('./models/sports1M_weights_tf.h5')
