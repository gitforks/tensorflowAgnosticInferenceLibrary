cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 11)

include_directories( 
	${OpenCV_INCLUDE_DIRS}, 
	"/home/jan/projects/tensorflowAgnosticInferenceLibrary/src/tf_inference_lib"
       	)

link_directories(
	"/home/jan/projects/tensorflowAgnosticInferenceLibrary/build_env/bin_cpu"
	)

add_executable( ObjectDetection ObjectDetection.cpp )

target_link_libraries( ObjectDetection ${OpenCV_LIBS} tf_inference_lib) # tensorflow_cc)
