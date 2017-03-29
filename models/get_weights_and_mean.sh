#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ---------------------------------------------
echo Downloading Sports1mil pre-trained model...
wget \
  -N \
  --content-disposition \
  http://vlg.cs.dartmouth.edu/c3d/conv3d_deepnetA_sport1m_iter_1900000 \
  --directory-prefix=${DIR}

echo ---------------------------------------------
echo Unpacking mean cube...
cp data/train01_16_128_171_mean.npy.bz2 ${DIR}
bunzip2 ${DIR}/train01_16_128_171_mean.npy.bz2

echo ---------------------------------------------
echo Done!
