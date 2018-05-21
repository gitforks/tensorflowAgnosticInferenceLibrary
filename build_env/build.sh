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

bazel build --config=opt --config=cuda --config=monolithic //tensorflow/examples/tf_inference_lib:libtf_inference_lib.so
cp bazel-bin/tensorflow/examples/tf_inference_lib/libtf_inference_lib.so $BIN_DST_DIR
cp tensorflow/examples/tf_inference_lib/*.h $BIN_DST_DIR

bazel build --config=opt --config=cuda --config=monolithic //tensorflow/contrib/tensorrt:python/ops/_trt_engine_op.so
cp bazel-bin/tensorflow/contrib/tensorrt/python/ops/_trt_engine_op.so $BIN_DST_DIR

bazel build --config=opt --config=cuda --config=monolithic //tensorflow/tools/graph_transforms:transform_graph
cp bazel-bin/tensorflow/tools/graph_transforms/transform_graph $BIN_DST_DIR

bazel build --config=opt --config=cuda --config=monolithic //tensorflow/tools/graph_transforms:summarize_graph
cp bazel-bin/tensorflow/tools/graph_transforms/summarize_graph $BIN_DST_DIR

bazel build --config=opt --config=cuda --config=monolithic //tensorflow/tools/graph_transforms:compare_graphs
cp bazel-bin/tensorflow/tools/graph_transforms/compare_graphs $BIN_DST_DIR

bazel build --config=opt --config=cuda --config=monolithic //tensorflow/tools/benchmark:benchmark_model
cp bazel-bin/tensorflow/tools/benchmark/benchmark_model $BIN_DST_DIR

bazel build --config=opt --config=monolithic //tensorflow:libtensorflow_cc.so
cp bazel-bin/tensorflow/libtensorflow_*.so $BIN_DST_DIR

