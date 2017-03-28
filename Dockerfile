FROM tensorflow/tensorflow:0.12.0

RUN \
  apt-get update; \
  apt-get install -y protobuf-compiler wget libcv-dev python-opencv; \
  apt-get install -y python-tk graphviz; \
  mkdir c3d-keras

ADD requirements.txt c3d-keras/requirements.txt
RUN pip install -r c3d-keras/requirements.txt

ADD models c3d-keras/models
ADD sports1m c3d-keras/sports1m
ADD data c3d-keras/data

RUN \
  cd c3d-keras; \
  bash models/get_weights_and_mean.sh; \
  bash sports1m/get_labels.sh; \
  wget -N https://raw.githubusercontent.com/facebook/C3D/master/C3D-v1.0/src/caffe/proto/caffe.proto; \
  protoc --python_out=. caffe.proto

ADD convert_caffe_model.py c3d-keras/convert_caffe_model.py
ADD c3d_model.py c3d-keras/c3d_model.py

RUN \
  cd c3d-keras; \
  echo import keras | python; \
  python convert_caffe_model.py

ADD download_test_video.sh c3d-keras/download_test_video.sh
ADD test_model.py c3d-keras/test_model.py
ADD dev_requirements.txt c3d-keras/dev_requirements.txt
RUN pip install -r c3d-keras/dev_requirements.txt

RUN \
  cd c3d-keras; \
  bash download_test_video.sh; \
  python test_model.py
