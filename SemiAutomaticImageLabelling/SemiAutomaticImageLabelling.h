#ifndef __SEMI_AUTOMATIC_IMAGE_LABELLING__
#define __SEMI_AUTOMATIC_IMAGE_LABELLING__

#include <vector>
#include "dataset.h"
#include "classifier.h"

//typedef ::std::pair<::std::string, int> TrainingSample;

struct TrainingSample
{
	::std::string name;

	::std::vector<int> features;

	int label;

	TrainingSample(::std::string name, ::std::vector<int> features, int label)
	{
		this->name = name;
		this->features = features;
		this->label = label;
	};

};

class ::cv::Mat;

class ImageLabelWorker
{
		int numOfLabels;

		::std::vector<TrainingSample> trainingSamples;

		Dataset data;

		::std::string outputPath;

		BagOfFeatures	bof;

	public:

		ImageLabelWorker();
		
		ImageLabelWorker(const ::std::string& path_to_images, const ::std::string& path_to_classifier);

		~ImageLabelWorker();

		void loadImages(const ::std::string& path_to_images);

		void extractFeaturesFromImages();

		void clusterImages();

		void labelClusters();

		void run();

		void writeToDisk();
		
	private:

		void _extractFeatures(::cv::Mat img, ::std::vector<int> response);

};

#endif