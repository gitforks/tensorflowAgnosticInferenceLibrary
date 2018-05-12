#/bin/bash

BIN_DST_DIR=/home/tf_bin/

mkdir /home/tf && cd /home/tf

git clone https://github.com/tensorflow/tensorflow 

cd tensorflow
git checkout v1.8.0

mkdir /home/tf_if && cd /home/tf_if

git clone https://github.com/x42x64/tensorflowAgnosticInferenceLibrary .

cp -R src/* /home/tf/tensorflow/tensorflow/examples/

cd /home/tf/tensorflow

./configure

bazel build --config=opt --config=monolithic //tensorflow/examples/tf_inference_lib:tf_inference_lib
cp bazel-bin/tensorflow/examples/tf_inference_lib/libtf_inference_lib.so $BIN_DST_DIR

bazel build --config=opt --config=monolithic //tensorflow:libtensorflow_cc.so
cp bazel-bin/tensorflow/libtensorflow_*.so $BIN_DST_DIR

