#!/usr/bin/env python

import c3d_model
import caffe_pb2 as caffe
import numpy as np
import h5py
import os

def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for k in range(W.shape[2]):
                W[i, j, k] = np.rot90(W[i, j, k], 2)
    return W

def main():
    # get C3D model placeholder
    model = c3d_model.get_model(summary=True)

    caffe_model_filename = '/home/chuck/projects/c3d-tensorflow2/models/conv3d_deepnetA_sport1m_iter_1900000'
    model_dir = './models'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_model_filename = os.path.join(model_dir, 'sports1M_weights.h5')
    output_json_filename = os.path.join(model_dir, 'sports1M_weights.json')

    print "-" * 19
    print "Reading model file={}...".format(caffe_model_filename)
    p = caffe.NetParameter()
    p.ParseFromString(open(caffe_model_filename, 'rb').read())

    params = []
    conv_layers_indx = [1, 4, 7, 9, 12, 14, 17, 19]
    fc_layers_indx = [22, 25, 28]

    print "-" * 19
    print "Converting model..."

    for i in conv_layers_indx:
        layer = p.layers[i]
        weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
        weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
            layer.blobs[0].height, layer.blobs[0].width
        )
        weights_p = rot90(weights_p)
        params.append([weights_p, weights_b])
    for i in fc_layers_indx:
        layer = p.layers[i]
        weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
        weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
            layer.blobs[0].height, layer.blobs[0].width)[0,0,0,:,:].T
        params.append([weights_p, weights_b])

    model_layers_indx = [0, 2, 4, 5, 7, 8, 10, 11] + [15, 17, 19] #conv + fc
    num_layers = len(model_layers_indx)
    for i, j in zip(model_layers_indx, range(num_layers)):
        model.layers[i].set_weights(params[j])

    print "-" * 19
    print "Saving pre-trained model weights as {}...".format(output_model_filename)
    model.save_weights(output_model_filename, overwrite=True)
    json_string = model.to_json()
    with open(output_json_filename, 'w') as f:
        f.write(json_string)
    print "-" * 39
    print "Conversion done!"
    print "-" * 39

if __name__ == '__main__':
    main()
