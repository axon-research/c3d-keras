C3D Model for Keras + TensorFlow
================================

The scripts here accompany [`C3D Model for Keras`](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2) gist, but specifically for Keras + TensorFlow (not Theano-backend).

Steps to reproduce results:

1. Download pretrained model: `bash models/get_weights_and_mean.sh`
2. Download sport1mil labels: `bash sports1m/get_labels.sh`
3. Download facebook/C3D `caffe.proto` file for conversion from caffe to Keras: `wget https://raw.githubusercontent.com/facebook/C3D/master/src/caffe/proto/caffe.proto`
4. Get protobuf that supports large enough `kDefaultTotalBytesLimit` as instructed in the [original gist](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2). Version 3.0 or later should support this.
5. Compile the caffe.proto file for python: `protoc --python_out=. caffe.proto`
6. Convert the pre-trained model from Caffe format to Keras: `python convert_caffe_model.py`
7. Run test: `python test_model.py`

References
==========

1. [facebook/C3D](https://arxiv.org/abs/1412.0767)
