set(tf_inference_lib_srcs
    "${tensorflow_source_dir}/tensorflow/examples/tf_inference_lib/main.cc"
)

set(tf_inference_lib_libsrcs
    "${tensorflow_source_dir}/tensorflow/examples/tf_inference_lib/cTfInference.cpp"
    "${tensorflow_source_dir}/tensorflow/examples/tf_inference_lib/cInferenceEngineFactory.cpp"    
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # this is needed, otherwise ops are not found

include_directories("${tensorflow_source_dir}/tensorflow/examples/tf_inference_lib/")

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native ")

add_library(TfInferenceCPU SHARED
	${tf_inference_lib_libsrcs}
	$<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_cc_framework>
        $<TARGET_OBJECTS:tf_core_cpu>
        $<TARGET_OBJECTS:tf_core_framework>
	$<TARGET_OBJECTS:tf_core_ops>
    $<TARGET_OBJECTS:tf_cc_ops>
    $<TARGET_OBJECTS:tf_core_kernels>
    $<TARGET_OBJECTS:tf_core_direct_session>
    #$<$<BOOL:${tensorflow_ENABLE_GPU}>:$<TARGET_OBJECTS:tf_stream_executor>>
)


target_link_libraries(TfInferenceCPU PUBLIC
    tf_protos_cc
    ${tf_core_gpu_kernels_lib}
    ${tensorflow_EXTERNAL_LIBRARIES}
)
   

#add_library(tf_inference_engine SHARED
#	    
#
#)



add_executable(test_inference_engine
    ${tf_inference_lib_srcs}
)

target_link_libraries(test_inference_engine
    TfInferenceCPU
)
