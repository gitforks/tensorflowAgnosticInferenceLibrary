cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )

set (CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
include_directories("/home/jan/tensorflow_cpp/cpu_only/tensorflow/tensorflow/examples/tf_inference_lib")
link_directories("/home/jan/tensorflow_cpp/deliverables/tensorflow1_0/gpu/bazel")
#link_directories("/home/jan/tensorflow_cpp/cpu_only/tensorflow/tensorflow/contrib/cmake/build")
add_executable( testOpenCV_tensorflow testOpenCV_tensorflow.cpp)

MESSAGE(STATUS "${OpenCV_LIBS}")
target_link_libraries( testOpenCV_tensorflow ${OpenCV_LIBS} libTfInferenceCPU.so  )
