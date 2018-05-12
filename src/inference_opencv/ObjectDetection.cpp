#include "opencv2/opencv.hpp"
#include "IInferenceEngine.h"

using namespace cv;

#define CAM_F (0.003584)
#define CAM_PX_X (3.0e-6)
#define CAM_PX_Y (3.0e-6)
#define CAM_CENTER_X (965.9097)
#define CAM_CENTER_Y (628.9913)
#define CAM_K1 (-0.333600)
#define CAM_K2 (0.084000)

#define MAX_NUM_DETECTIONS 32

int main(int, char**)
{
    VideoCapture cap("/home/jan/Videos/sekonix_100.h264"); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    
    tf_interface_lib::IInferenceEngine* pInferenceEngine =
		 tf_interface_lib::cInferenceEngineFactory::getInferenceEngine();

    pInferenceEngine->init("/home/jan/projects/opencv_tf_chain/faster_rcnn_kitti.pb");

    pInferenceEngine->addInput("image_tensor", {1, 600, 1800, 3}, tf_interface_lib::eExchangeDataType::DT_UINT8);

    pInferenceEngine->addOutput("num_detections", {1}, tf_interface_lib::eExchangeDataType::DT_FLOAT);
    pInferenceEngine->addOutput("detection_scores", {MAX_NUM_DETECTIONS, 1}, tf_interface_lib::eExchangeDataType::DT_FLOAT);
    pInferenceEngine->addOutput("detection_boxes", {MAX_NUM_DETECTIONS, 4}, tf_interface_lib::eExchangeDataType::DT_FLOAT);
    pInferenceEngine->addOutput("detection_classes", {MAX_NUM_DETECTIONS, 1}, tf_interface_lib::eExchangeDataType::DT_FLOAT);

    pInferenceEngine->infer();


    Mat undistortedImg;
    Mat cropped;
    Mat inputImg(Size(1800, 600), CV_8UC3, pInferenceEngine->getInputData(0));
    Mat resized;
    Mat markedImg;

    Mat camMatrix = Mat::eye(3,3,CV_32F);
    camMatrix.at<float>(0,0) = CAM_F / CAM_PX_X;
    camMatrix.at<float>(1,1) = CAM_F / CAM_PX_Y;
    camMatrix.at<float>(0,2) = CAM_CENTER_X;
    camMatrix.at<float>(1,2) = CAM_CENTER_Y;

    Mat distortionCoeff = (Mat_<float>(4,1) << CAM_K1, CAM_K2, 0.0, 0.0);
    Mat undistortMap1;
    Mat undistortMap2;
    
    initUndistortRectifyMap(camMatrix, distortionCoeff, Mat::eye(3,3,CV_32F), camMatrix, Size(1980, 1208), CV_16SC2, undistortMap1, undistortMap2);

    for(int l=0;;++l)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        //cvtColor(frame, edges, COLOR_BGR2GRAY);
        //GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        //Canny(edges, edges, 0, 30, 3);
	
	remap(frame, undistortedImg, undistortMap1, undistortMap2, INTER_LINEAR);
//	imshow("undistorted", undistortedImg);
        //undistortedImg = frame;
        Rect roi;
	roi.x = 3*45;
	roi.y = 330 + 3*50;
	roi.width = 3*450;
	roi.height = 3*150;

	cropped = undistortedImg(roi);
	resize(cropped, inputImg, Size(1800, 600));
//	resized.convertTo(inputImg, CV_32F);

//        inputImg = ((inputImg / 255.0) );
//	cvtColor(inputImg, inputImg, CV_RGB2BGR);

	if(l%90==0)
	{
		pInferenceEngine->infer();
    		float num_detections = *(float*)pInferenceEngine->getOutputData(0);
	
 	      	//std::cout << num_detections << std::endl;
		Mat scores(MAX_NUM_DETECTIONS, 1, CV_32FC1, pInferenceEngine->getOutputData(1));
		Mat boxes(MAX_NUM_DETECTIONS, 4, CV_32FC1, pInferenceEngine->getOutputData(2));
		Mat classId(MAX_NUM_DETECTIONS, 1, CV_32FC1, pInferenceEngine->getOutputData(3));

		for (uint8_t i=0; i<(int)num_detections; i++)
		{
			std::cout << "Class: " << classId.at<float>(i, 0) << " with score " << scores.at<float>(i,0) << std::endl;
			int x = (int) (boxes.at<float>(i,1) * 1800.0);
			int y = (int) (boxes.at<float>(i,0) * 600.0);
			int right = (int) (boxes.at<float>(i,3) * 1800.0);
			int bottom = (int) (boxes.at<float>(i,2) * 600.0);

			rectangle(inputImg, Point(x,y), Point(right,bottom), Scalar(255, 0, 0), 3);
        		imshow("org", inputImg);

		}
	}

        if(waitKey(03) == 'q') break;

    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

