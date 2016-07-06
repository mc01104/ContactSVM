// trainSVM.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <Windows.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>
#include "Utilities.h"

void ExampleSVM();
void LoadTrainingData(::std::vector<::std::string>& listOfFiles);
void LabelData(const ::std::vector<::std::string>& listOfFiles, cv::Mat& labels);
void FeatureExtraction(const ::std::vector<::std::string>& listOfFiles, ::Eigen::MatrixXd& featureMatrix);
void FeatureExtraction(const ::std::vector<::std::string>& listOfFiles, ::cv::Mat& featureMatrix);

::std::string searchpath = "C:/Users/RC/Dropbox/Boston/BCH/surgery/2016-05-26_14-14-40/*.png";
::std::string path = "C:/Users/RC/Dropbox/Boston/BCH/surgery/2016-05-26_14-14-40/";

int _tmain(int argc, _TCHAR* argv[])
{
	// load data
	::std::vector<::std::string> listOfFiles;
	LoadTrainingData(listOfFiles);
	::std::cout << "Loaded " << listOfFiles.size() << " files!!" << ::std::endl;

	// feature extraction
	//int numOfFeatures = 1;
	//::Eigen::MatrixXd featureMatrix(listOfFiles.size(), numOfFeatures);
	//FeatureExtraction(listOfFiles, featureMatrix);

	int numOfFeatures = 1;	
	::cv::Mat featureMatrix(listOfFiles.size(), numOfFeatures, CV_32F);
	FeatureExtraction(listOfFiles, featureMatrix);


	// label data
	cv::Mat labelsMat;
	int numberOfPcsToLabel = 5;
	::std::vector<::std::string> shortListOfFiles(listOfFiles.begin(), listOfFiles.begin() + numberOfPcsToLabel);
	LabelData(shortListOfFiles, labelsMat);

	
	// train classifier
	//::cv::Ptr<::cv::ml::SVM> svm = ::cv::ml::SVM::create();
 //   svm->setType(::cv::ml::SVM::C_SVC);
 //   svm->setKernel(::cv::ml::SVM::LINEAR);
 //   svm->setTermCriteria(::cv::TermCriteria::TermCriteria(::cv::TermCriteria::TermCriteria::MAX_ITER, 100, 1e-6));
 //   svm->train(featureMatrix, ::cv::ml::SampleTypes::ROW_SAMPLE, labelsMat);
	//// test classifier

	// save classifier
	return 0;
}

void FeatureExtraction(const ::std::vector<::std::string>& listOfFiles, ::Eigen::MatrixXd& featureMatrix)
{
	cv::namedWindow("Image", CV_WINDOW_AUTOSIZE );
	cv::RNG rng;
	int counter = 0;

	for (::std::vector<::std::string>::const_iterator it = listOfFiles.begin(); it != listOfFiles.end(); ++it)
	{
		cv::Mat image, imageRGB;
		vector<cv::Point2f> corners;
		imageRGB = cv::imread(path + it->c_str(), cv::IMREAD_COLOR); // Read the file
		image = cv::imread(path + it->c_str(), cv::IMREAD_GRAYSCALE); // Read the file
 
		// Define ROI
		cv::Rect Rec(80, 0, 100, 250);
		cv::rectangle(imageRGB, Rec, cv::Scalar(255), 1, 8, 0);
		cv::Mat mask = cv::Mat::zeros(250, 250, CV_8U); // all 0
		mask(Rec) = 1;

		cv::goodFeaturesToTrack(image, corners, 30, 0.1, 10,  mask, 3, false);

		featureMatrix(counter++, 0) = corners.size();

		int r = 4;
		for( int i = 0; i < corners.size(); i++ )
			cv::circle( imageRGB, corners[i], r, cv::Scalar(0, 255, 0), -1, 8, 0 ); 

	    /// Show what you got
		cv::imshow("Image", imageRGB);
		cv::waitKey(0);
	}
	
}


void FeatureExtraction(const ::std::vector<::std::string>& listOfFiles, ::cv::Mat& featureMatrix)
{
	cv::namedWindow("Image", CV_WINDOW_AUTOSIZE );
	cv::RNG rng;
	int counter = 0;

	for (::std::vector<::std::string>::const_iterator it = listOfFiles.begin(); it != listOfFiles.end(); ++it)
	{
		cv::Mat image, imageRGB;
		vector<cv::Point2f> corners;
		imageRGB = cv::imread(path + it->c_str(), cv::IMREAD_COLOR); // Read the file
		image = cv::imread(path + it->c_str(), cv::IMREAD_GRAYSCALE); // Read the file
 
		// Define ROI
		cv::Rect Rec(80, 0, 100, 250);
		cv::rectangle(imageRGB, Rec, cv::Scalar(255), 1, 8, 0);
		cv::Mat mask = cv::Mat::zeros(250, 250, CV_8U); // all 0
		mask(Rec) = 1;

		cv::goodFeaturesToTrack(image, corners, 30, 0.1, 10,  mask, 3, false);

		featureMatrix.at<float>(counter++, 0) = static_cast<float> (corners.size());

		int r = 4;

		for( int i = 0; i < corners.size(); i++ )
			cv::circle( imageRGB, corners[i], r, cv::Scalar(0, 255, 0), -1, 8, 0 ); 

		cv::imshow("Image", imageRGB);
		cv::waitKey(0);
	}
	
}


void LoadTrainingData(::std::vector<::std::string>& listOfFiles)
{
	HANDLE hFind;
	WIN32_FIND_DATA data;
	::std::wstring path_w = s2ws(searchpath);

	hFind = FindFirstFile(path_w.c_str(), &data);
	if (hFind != INVALID_HANDLE_VALUE) {
	  do {
		  ::std::wstring tmpWStr = data.cFileName;
		  ::std::string tmpStr(tmpWStr.begin(), tmpWStr.end());
		  listOfFiles.push_back(tmpStr);

	  } while (FindNextFile(hFind, &data));
	  FindClose(hFind);
	}
}


void LabelData(const ::std::vector<::std::string>& listOfFiles, cv::Mat& labelsMat)
{
	cv::namedWindow("Image", CV_WINDOW_AUTOSIZE );
	cv::Mat imageRGB;
	int counter = 0;
	int numOfTrainingSamples = listOfFiles.size();
	
	for (::std::vector<::std::string>::const_iterator it = listOfFiles.begin(); it != listOfFiles.end(); ++it)
	{		
		imageRGB = cv::imread(path + it->c_str(), cv::IMREAD_COLOR); // Read the file
		cv::imshow("Image", imageRGB);
		int keyPressed = cv::waitKey(0);
		
		// contact
		if (keyPressed == 99)
			labelsMat.push_back<int>(1);
		// contact-free
		else if (keyPressed == 102)
			labelsMat.push_back<int>(0);

		counter++;
	}

}


void ExampleSVM()
{
	using namespace cv;
	using namespace cv::ml;

    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);
    // Set up training data
    int labels[4] = {1, -1, -1, -1};
    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);
    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    // Show the decision regions given by the SVM
    Vec3b green(0,255,0), blue (255,0,0);
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
        }
    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType );
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType );
    // Show support vectors
    thickness = 2;
    lineType  = 8;
    Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }
    imwrite("result.png", image);        // save the image
    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);

}
