#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

// general includes
#include <vector>
#include <string>

// opencv
#include <opencv2\opencv.hpp>

/**
  *	@brief: interface class for image classification
  * @author: George & Ben
  */

class ImageClassifier
{
		// attributes
		::std::vector<::std::string> m_classes;

	public:
		
		// interface
		ImageClassifier();

		~ImageClassifier();

		virtual bool predict(const ::std::vector<::cv::Mat*>& imgs, ::std::vector<float>& labels) const;

		::std::vector<::std::string> getClasses()  const {return m_classes;};
		
		// pure virtual functions - MUST be implemented in every subclass in the inheritance tree
		virtual bool load(const ::std::string& path_to_classifier_files) = 0;

		virtual bool save(const ::std::string& path_to_classifier_files) = 0;
		
		virtual bool train(::std::vector<::cv::Mat*> imgs, ::std::vector<float>& labels) = 0;

		virtual bool predict(const ::cv::Mat* const img, float& response) const = 0 ;
		
};

/**
  *	@brief: interface class for image classification
  * @author: George & Ben
  */

class BagOfFeatures : public ImageClassifier
{
	// attributes
	protected:
		::cv::Ptr<::cv::FeatureDetector> m_featureDetector;
		::cv::Ptr<::cv::DescriptorExtractor> m_descriptorExtractor;
		::cv::Ptr<::cv::DescriptorMatcher> m_descriptorMatcher;

		::cv::Mat m_vocabulary;


		::cv::Ptr<::cv::ml::SVM> m_svm ;
		::cv::TermCriteria m_tc_Kmeans;
		::cv::TermCriteria m_tc_SVM;

		::cv::Ptr<::cv::ml::KNearest> m_knn;

		std::vector<std::string> m_classes;
		std::vector<std::string> m_imList;
		std::vector<std::string> m_classList;
	
		std::vector<float> m_scaling_means;
		std::vector<float> m_scaling_stds;

		int m_dictionarySize;

		bool m_trained;

	// interface
	public:
		BagOfFeatures();

		virtual ~BagOfFeatures();

		virtual bool load(const ::std::string& path_to_classifier_files);

		virtual bool save(const ::std::string& path_to_classifier_files);

		virtual bool train(const ::std::vector<::cv::Mat*> imgs, ::std::vector<float>& labels) = 0;

		virtual bool predict(const ::cv::Mat* const img, float& response) const = 0;

	// implementation
	protected:

		void initializeKNN(::cv::ml::KNearest::Types KNNSearchDataStructure = ::cv::ml::KNearest::KDTREE);

		void featureExtraction(const ::std::vector<::cv::Mat*>& imgs, ::std::vector<int>& image_number, ::cv::Mat& training_descriptors);

		void computeResponseHistogram(const ::std::vector<::cv::Mat*> imgs, const ::cv::Mat& cluster_labels, const ::std::vector<int>& image_number, ::cv::Mat& im_histograms);
};


#endif	__CLASSIFIER_H__