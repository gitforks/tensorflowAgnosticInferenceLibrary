#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "IInferenceEngine.h"

using namespace cv;
using namespace tf_interface_lib;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    Mat image_resized;
    Mat fImage;
    image = imread( argv[1], 1 );
    printf("imagepath %s \n", argv[1]);

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    resize(image, image_resized, Size(299,299), 0, 0, INTER_LINEAR);
    cvtColor(image_resized, image, CV_RGB2BGR);
    image.convertTo(fImage, CV_32F);
    fImage = (fImage - 128.0)/128.0;

    fImage = fImage.clone();
    printf("Image was not continous? now it is %i", fImage.isContinuous());



    tf_interface_lib::IInferenceEngine* pInferenceEngine = tf_interface_lib::cInferenceEngineFactory::getInferenceEngine();

    pInferenceEngine->init(
    		"/home/jan/tensorflow_cpp/cpu_only/tensorflow/tensorflow/examples/tf_inference_lib/data/tensorflow_inception_graph.pb"
    		);

    tensor_exchange_t exInput[1];
    pInferenceEngine->addInput("Mul", {1, 299, 299, 3}, eExchangeDataType::DT_FLOAT, exInput[0]);

    tensor_exchange_t exOutput[1];
    pInferenceEngine->addOutput("softmax", {1, 1008}, eExchangeDataType::DT_FLOAT, exOutput[0]);


    float* startImage = fImage.ptr<float>(0);
    float* secondRow = startImage + fImage.channels()*(fImage.cols*1);
    float* secondRow2 = fImage.ptr<float>(1);
    if(secondRow != secondRow2)
    {
    	printf("\n There seems to be padding. %p, %p \n", secondRow, secondRow2);
    }



    exInput[0].mem = fImage.ptr<float>(0);
    exInput[0].data_len = 299*299*3*sizeof(float);


    float* inpIt = (float*)exInput[0].mem;
    for(int i=0; i<299*299*3; i++)
    {
    	if(inpIt == nullptr)
    		break;

    	if((*inpIt > 1.0 || *inpIt < -1.0))
    	{
    		printf("exceeding Value at %i: %f\n", i, *inpIt);
    	}
    	if(i<100)
    	{
    		printf("%f \n", *inpIt);
    	}
    	inpIt++;
    }


    for (int m = 0; m<100; m++)
    {
		pInferenceEngine->infer(exInput, 1, exOutput, 1);

		float* it = (float*) exOutput[0].mem;
		float max_val = 0.0;
		int max_idx = -1;
		for(int i= 0; i< 1008; i++)
		{
			if(it == nullptr)
				break;

			if(*it > max_val)
			{
				max_val = *it;
				max_idx = i;
			}
			it++;
		}

		printf("Best class: %i @ %f \n", max_idx, max_val);

    }

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", fImage);

    //waitKey(0);


    //free(contMem);
    delete pInferenceEngine;

    return 0;
}
