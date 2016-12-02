#!/usr/bin/env python

import c3d_model
import caffe_pb2 as caffe
import numpy as np
import h5py
import os

# why needed?
def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[3]):
            for k in range(W.shape[4]):
                W[i, :, :, j, k] = np.rot90(W[i, :, :, j, k], 2)
    return W

def main():

    # get C3D model placeholder
    model = c3d_model.get_model(summary=True)

    # input caffe model
    caffe_model_filename = '/home/chuck/projects/c3d-tensorflow2/models/conv3d_deepnetA_sport1m_iter_1900000'

    # output dir/files
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_model_filename = os.path.join(model_dir, 'sports1M_weights.h5')
    output_json_filename = os.path.join(model_dir, 'sports1M_weights.json')

    # read caffe model
    print "-" * 19
    print "Reading model file={}...".format(caffe_model_filename)
    p = caffe.NetParameter()
    p.ParseFromString(open(caffe_model_filename, 'rb').read())

    params = []
    print "-" * 19
    print "Converting model..."

    # read every conv/fc layer and append to "params" list
    for i in range(len(p.layers)):
        layer = p.layers[i]
        # skip non-conv/fc layers
        if 'conv' not in layer.name and 'fc' not in layer.name:
            continue
        print "[Info] Massaging \"{}\" layer...".format(layer.name)
        weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
        weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num,
            layer.blobs[0].channels,
            layer.blobs[0].length,
            layer.blobs[0].height,
            layer.blobs[0].width,
            )
        weights_p = np.transpose(weights_p, (2, 4, 3, 1, 0))
        if 'conv' in layer.name:
            #pass
            # why rot90 twice?
            weights_p = rot90(weights_p)
        elif 'fc' in layer.name:
            #weights_p = weights_p[0, 0, 0, :, :].T
            weights_p = np.squeeze(weights_p)
        params.append([weights_p, weights_b])

    valid_layer_count = 0
    for layer_indx in range(len(model.layers)):
        layer_name = model.layers[layer_indx].name
        if 'conv' in layer_name or 'fc' in layer_name:
            print "[Info] Transplanting \"{}\" layer...".format(layer_name)
            model.layers[layer_indx].set_weights(params[valid_layer_count])
            valid_layer_count += 1

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
