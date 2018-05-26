cmake_minimum_required(VERSION 3.0.0)

project(tf_inference_lib VERSION 1.0.0 DESCRIPTION "tensorflow inference lib")

set(TENSORFLOW_DIR xxxx)

include_directories(
    "${TENSORFLOW_DIR}/include"
)

link_directories(
    "${TENSORFLOW_DIR}/lib"
)

file(GLOB SRC
    "*.cpp"
    "*.h"
    "*.hpp"
)

add_library(${PROJECT_NAME} SHARED ${SRC})


target_link_libraries(${PROJECT_NAME}
     tensorflow_cc
)
