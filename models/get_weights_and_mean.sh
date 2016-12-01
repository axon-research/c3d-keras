#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ---------------------------------------------
echo Downloading Sports1mil pre-trained model...
wget \
  -nc \
  --content-disposition \
  http://vlg.cs.dartmouth.edu/c3d/conv3d_deepnetA_sport1m_iter_1900000 \
  --directory-prefix=${DIR}

echo ---------------------------------------------
echo Downloading mean cube...
wget \
  -nc \
  https://github.com/facebook/C3D/raw/master/examples/c3d_finetuning/train01_16_128_171_mean.binaryproto \
  --directory-prefix=${DIR}

echo ---------------------------------------------
echo Done!
