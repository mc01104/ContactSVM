#include "stdafx.h"
#include "SemiAutomaticImageLabelling.h"

#include <opencv2/opencv.hpp>



ImageLabelWorker::ImageLabelWorker():
	numOfLabels(2)
{
}

ImageLabelWorker::ImageLabelWorker(const ::std::string& path_to_images)
{
	data.initDataset(path_to_images);
}

ImageLabelWorker::~ImageLabelWorker()
{
}


void ImageLabelWorker::extractFeaturesFromImages()
{
	//::std::vector<::cv::Mat> imgs = data.getImages();
	::std::vector<::std::string> imList = data.getImagesList();

	::std::vector<int> featureVector;
	::cv::Mat tmpImg;
	for(int i = 0; i < imList.size(); ++i)
	{
		tmpImg = ::cv::imread(imList[i]);
		this->_extractFeatures(tmpImg, featureVector);
		this->trainingSamples.push_back(TrainingSample(imList[i], featureVector, 0));
	}

}

void ImageLabelWorker::labelClusters()
{
}

void ImageLabelWorker::run()
{
	this->clusterImages();

	this->labelClusters();

	this->writeToDisk();
}

void ImageLabelWorker::loadImages(const ::std::string& path_to_images)
{
	if (this->data.isInit)
		this->data.clear();

	this->data.initDataset(path_to_images);

}

void ImageLabelWorker::_extractFeatures(::cv::Mat img, ::std::vector<int> response)
{

	std::vector<cv::KeyPoint> keyPoints;
	::cv::Mat descriptors;

	::cv::Ptr< ::cv::FeatureDetector> featureDetector;
	::cv::Ptr< ::cv::DescriptorExtractor> descriptorExtractor;
	::cv::Ptr< ::cv::DescriptorMatcher> descriptorMatcher;

	// detect keypoints
	featureDetector->detect(img, keyPoints);

	// extract descriptors
	descriptorExtractor->compute(img, keyPoints, descriptors);

	//training_descriptors.push_back(descriptors);

	//	for (int j = 0; j< descriptors.rows; j++)
	//		image_number.push_back(i);
	//}

	//// put descriptors in right format for kmeans clustering
	//if(training_descriptors.type()!=CV_32F) 
	//	training_descriptors.convertTo(training_descriptors, CV_32F); 
}