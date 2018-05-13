# tensorflowAgnosticInferenceLibrary

This is a wrapper library in order to use tensorflow frozen graphs without having to compile or handle any specific tensorflow code.

This repository consists of 3 parts:
* a [Dockerfile](build_env/Dockerfile) and a [script](build_env/build.sh) to build this library
* the [sources](src/tf_inference_lib) for the library
* an [example project](src/inference_opencv) which uses openCV to read and prepare an image and then uses the library to do inference on it.

## How to build the library

1. Build the docker container: `docker build -t x42x64/tf_build <path to build_env>`
2. create a folder where the resulting binaries should be located
3. start a temporary docker container and mount the result binary folder accordingly: `nvidia-docker run --rm -it -v <my path to the resulting binaries>:/home/tf_bin x42x64/tf_build /bin/bash`

> The `nvidia-` part of `nvidia-docker` is not necessary if you are building for CPU only.
> If it is necessary for building with GPU support, I don't know. I suspect no, but on the other hand all automatic detection of the compute capabilities and so on will not work

4. within the docker container, start the build.sh script: `/root/build.sh`
5. You will be prompted, on how to build tensorflow. Select your options accordingly (GPU support etc.)
6. After the build finished, a >100MB shared object called `libtf_inference_lib.so` should be present in the mounted folder.
