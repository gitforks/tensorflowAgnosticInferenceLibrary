# tensorflowAgnosticInferenceLibrary

This is a wrapper library in order to use tensorflow frozen graphs without having to compile or handle any specific tensorflow code.

This repository consists of 3 parts:
* a [Dockerfile](build_env/Dockerfile) and a [script](build_env/build.sh) to build this library (but not the example as it needs opencv)
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

## Example usage

Here is a minimalist example with blanks for you to fill in:

```c++
#include "IInferenceEngine.h"

int main(int, char**)
{
    // create a handle to an inference engine
    tf_interface_lib::IInferenceEngine* pInferenceEngine =
		 tf_interface_lib::cInferenceEngineFactory::getInferenceEngine();

    // load the frozen graph and initializing the inference engine
    pInferenceEngine->init("/path/to/frozen/graph/faster_rcnn_kitti.pb");

    // specify an input for the network.
    // addInput(string name, int[] dimensions, dataType)
    pInferenceEngine->addInput("image_tensor", {1, 600, 1800, 3}, tf_interface_lib::eExchangeDataType::DT_UINT8);

    // specify multiple outputs of a network
    // addOutput(string name, int[] dimensions, dataType)
    pInferenceEngine->addOutput("num_detections", {1}, tf_interface_lib::eExchangeDataType::DT_FLOAT);

    // get the memory location where to put the input data
    uint8_t* inputData = pInferenceEngine->getInputData(0); // 0 because we want to have the location 
                                                            // of the first input which was registered.
                                                            // this pointer will not change after the input was added.
                                                            
    while(...)
    {
      // fill the inputData memory with proper data
      fillMyData(inputData);
      
      // do the inference
      pInferenceEngine->infer();
      
      // get the results
      float* numDetections = pInferenceEngine->getInputData(0); // this pointer might change after every inference!
      std::cout << "Number of detections: " << *numDetections << std::endl;
    }
  
  return 0;
}
```



