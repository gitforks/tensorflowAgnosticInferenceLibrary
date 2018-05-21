# Description:
#   TensorFlow C++ inference example for labeling images.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@local_config_tensorrt//:build_defs.bzl",
     "if_tensorrt",
)

cc_library(
	name = "tf_inference_lib",
	srcs = [
		"cTfInference.cpp",
        "cTfInference.hpp",
        "cInferenceEngineFactory.cpp",
        "IInferenceEngine.h",
	],
	deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        ],
)

cc_binary(
	name = "libtf_inference_lib.so",
	srcs = [
		"cTfInference.cpp",
        "cTfInference.hpp",
        "cInferenceEngineFactory.cpp",
        "IInferenceEngine.h",
	],
	deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/cc/profiler"
        ] + if_tensorrt([
        "//tensorflow/contrib/tensorrt:trt_engine_op_kernel",
        ]),
	linkopts=["-shared"],
	linkshared=1,
	visibility=["//visibility:public"]
)

cc_binary(
    name = "test_lib",
    srcs = [
        "main.cc",
    ],
    linkopts = ["-lm"],
    deps = [
    	":tf_inference_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
    ],
)

cc_binary(
    name = "libTfInferenceCPU.so",
    linkshared=1, 
    visibility=["//visibility:public"],
    deps = [
    	":tf_inference_lib",
        "//tensorflow/contrib/tensorrt:trt_engine_op_kernel",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow"
    ],
)


filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
