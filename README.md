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
7. Download test video: `bash download_test_video.sh`
8. Run test: `python test_model.py`

Results
=======
A following classification probability plot is expected. A peak at 367th class corresponds to `basketball` label.
![classificatino probability plot] (classification_probability.png?raw=true "Classification Probability Plot")
```
Position of maximum probability: 367
Maximum probability: 0.71422
Corresponding label: basketball

Top 5 probabilities and labels:
basketball: 0.71422
streetball: 0.10293
volleyball: 0.04900
greco-roman wrestling: 0.02638
freestyle wrestling: 0.02408
```

Note
======
- The first inner product (Dense) layer after Flatten layer needs to be massaged for TF dim-ordering.

References
==========

1. [Original C3D implementation in Caffe](https://github.com/facebook/C3D)
2. [C3D paper](https://arxiv.org/abs/1412.0767)
