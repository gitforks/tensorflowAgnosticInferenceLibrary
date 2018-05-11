#/bin/bash

mkdir /home/tf && cd /home/tf

git clone https://github.com/tensorflow/tensorflow 

cd tensorflow
git checkout v1.8.0

mkdir /home/tf_if && cd /home/tf_if

git clone https://github.com/x42x64/tensorflowAgnosticInferenceLibrary .

cp -R src/* /home/tf/tensorflow/tensorflow/examples/

cd /home/tf/tensorflow

./configure

