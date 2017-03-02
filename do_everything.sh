#!/usr/bin/env bash

# suppress some logs
export TF_CPP_MIN_LOG_LEVEL=1

# get weights / mean cube
bash models/get_weights_and_mean.sh

# get sports1mil labels
bash sports1m/get_labels.sh

# get caffe.proto from facebook/C3D repo (thanks to Du Tran)
wget -N https://raw.githubusercontent.com/facebook/C3D/master/C3D-v1.0/src/caffe/proto/caffe.proto

# protobuf combile caffe.proto
if [ "$(which protoc 2> /dev/null)" ]; then
  protoc --python_out=. caffe.proto
else
  echo 'Please install protobuf-compiler and rerun this script. e.g. "sudo apt-get install protobuf-compiler"'
  exit -1
fi

# make sure the default keras config (in `~/.keras/keras.json`) has: `tf` image_dim_ordering, and `tensorflow` backend.
KERASCONF=~/.keras/keras.json
if [ -z "$(grep image_dim_ordering.*tf ${KERASCONF})" ]; then
  echo 'Please set "image_dim_ordering" in ${KERASCONF} to be "tf"'
  exit -1
fi
if [ -z "$(grep backend.*tensorflow ${KERASCONF})" ]; then
  echo 'Please set "backend" in ${KERASCONF} to be "tensorflow"'
  exit -1
fi

# finally do the conversion!
python convert_caffe_model.py

# download test video (basketball clip)
bash download_test_video.sh

# run classification on this video
python test_model.py

echo "---------------------------------------------------"
echo "You should have something close to the following:"
echo "---------------------------------------------------"
echo "Position of maximum probability: 367"
echo "Maximum probability: 0.71422"
echo "Corresponding label: basketball"
echo ""
echo "Top 5 probabilities and labels:"
echo "basketball: 0.71422"
echo "streetball: 0.10293"
echo "volleyball: 0.04900"
echo "greco-roman wrestling: 0.02638"
echo "freestyle wrestling: 0.02408"
