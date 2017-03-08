#include "classifier.h"

#include <opencv2\xfeatures2d\nonfree.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2/ml.hpp>

#include "Utilities.h"
#include "FileUtils.h"

ImageClassifier::ImageClassifier()
{
}

ImageClassifier::~ImageClassifier()
{
}

bool 
ImageClassifier::predict(const ::std::vector<::cv::Mat*>& imgs, ::std::vector<float>& labels) const
{
	if (imgs.size() < 1)
		throw(::std::exception("Vector of images is empty!!"));

	float tmpResponse = 0.0;
	labels.resize(0);

	for (::std::vector<::cv::Mat*>::const_iterator it = imgs.begin(); it != imgs.end(); ++it)
	{
		if(!this->predict(*it, tmpResponse))
			return false;

		labels.push_back(tmpResponse);
	}

	return true;
}

//@TODO - supports only LUCID feature descirptors at the moment
BagOfFeatures::BagOfFeatures():
	ImageClassifier(),
	m_dictionarySize(0),
	m_trained(false)
{
	m_featureDetector =  cv::FastFeatureDetector::create();
	m_descriptorExtractor = cv::xfeatures2d::LUCID::create(2,1);

	m_tc_Kmeans = ::cv::TermCriteria(::cv::TermCriteria::MAX_ITER + ::cv::TermCriteria::EPS,100000, 0.000001);

}

BagOfFeatures::~BagOfFeatures()
{
}


bool BagOfFeatures::load(const ::std::string& path_to_classifier_files) 
{
	try
	{
		// Load vocabulary
		cv::FileStorage storage(path_to_classifier_files + "VOC.xml", cv::FileStorage::READ);
		storage["vocabulary"] >> m_vocabulary;
		storage.release();  

		// Load KNN if there is a file or train a new one otherwise
		if (file_exists(path_to_classifier_files + "KNN.xml"))
		{
			::cv::FileStorage readKNN(path_to_classifier_files +"KNN.xml", ::cv::FileStorage::READ);
			m_knn = ::cv::Algorithm::read<::cv::ml::KNearest>(readKNN.root());

			m_trained = true;	 // not sure if I need that
		}
		else
			this->initializeKNN();

		// Load SVM parameters
		m_svm = ::cv::ml::StatModel::load<::cv::ml::SVM>(path_to_classifier_files + "SVM.xml");

		// Ben is that correct???
		m_dictionarySize = m_vocabulary.rows;

		storage = ::cv::FileStorage(path_to_classifier_files + "SCALE_means.xml", cv::FileStorage::READ);
		storage["scale_m"] >> m_scaling_means;
		storage.release();   

		storage = ::cv::FileStorage(path_to_classifier_files + "SCALE_stds.xml", cv::FileStorage::READ);
		storage["scale_stds"] >> m_scaling_stds;
		storage.release();   

		m_classes.clear();
		storage = ::cv::FileStorage(path_to_classifier_files + "CLASSES.xml", cv::FileStorage::READ);
		::cv::FileNode n = storage["classes"];                         // Read string sequence - Get node
		::cv::FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
		for (; it != it_end; ++it)
			m_classes.push_back((::std::string)*it);
		storage.release();   

		m_trained = true;
		return true;
	}
	catch ( const std::exception & e ) 
	{
		::std::cerr << e.what();

		m_trained = false;
		return false;
	}

}

bool BagOfFeatures::save(const ::std::string& path_to_classifier_files)
{
	try
	{
		m_knn->save(path_to_classifier_files + "KNN.xml");
		
		m_svm->save(path_to_classifier_files + "SVM.xml");

		cv::FileStorage storage(path_to_classifier_files + "VOC.xml", cv::FileStorage::WRITE);
		storage << "vocabulary" << m_vocabulary;
		storage.release();   


		storage = ::cv::FileStorage(path_to_classifier_files + "SCALE_means.xml", cv::FileStorage::WRITE);
		storage << "scale_m" << m_scaling_means;
		storage.release();   

		storage = ::cv::FileStorage(path_to_classifier_files + "SCALE_stds.xml", cv::FileStorage::WRITE);
		storage << "scale_stds" << m_scaling_stds;
		storage.release();   

		storage = ::cv::FileStorage(path_to_classifier_files + "CLASSES.xml", cv::FileStorage::WRITE);
		storage << "classes" << "[";
		for (int i=0;i<m_classes.size();i++) storage<<m_classes[i];
		storage << "]"; 
		storage.release();   

		return true;
	}
	catch ( const std::exception & e ) 
	{
		::std::cerr << e.what();
		return false;
	}
}

bool BagOfFeatures::train(::std::vector<::cv::Mat*> imgs, ::std::vector<float>& labels)
{
	
	// feature extraction
	::std::vector<int> image_number; // this looks as if it could be done in a more elegant way
	::cv::Mat training_descriptors(0, m_descriptorExtractor->descriptorSize(), m_descriptorExtractor->descriptorType());
	featureExtraction(imgs, image_number, training_descriptors);

	// kmeans cluster to construct the vocabulary
	::cv::Mat cluster_labels;
	::cv::kmeans(training_descriptors, m_dictionarySize, cluster_labels, m_tc_Kmeans, 3, cv::KMEANS_PP_CENTERS, m_vocabulary );
	this->initializeKNN();

	// compute response histograms for all training images
	::cv::Mat im_histograms = ::cv::Mat::zeros(imgs.size(), m_dictionarySize, CV_32FC1);
	computeResponseHistogram(imgs, cluster_labels, image_number, im_histograms);

	// train SVM classifier
	m_svm = ::cv::ml::SVM::create();
	m_svm->setGamma(0.02);

	m_trained = (m_svm->train(im_histograms, ::cv::ml::ROW_SAMPLE,labels) ? true : false); 

	return m_trained;

}

bool BagOfFeatures::predict(const ::cv::Mat* const img, float& response) const
{
	if (!m_trained) 
		return false;

	std::vector<cv::KeyPoint> keyPoints;
	::cv::Mat descriptors;

	std::vector<int> v_word_labels;
	for (int i = 0; i < m_vocabulary.rows; ++i) 
		v_word_labels.push_back(i);

	m_featureDetector->detect(*img, keyPoints);
	m_descriptorExtractor->compute(*img, keyPoints,descriptors);
	
	// put descriptors in right format for kmeans clustering
	if(descriptors.type()!=CV_32F) 
		descriptors.convertTo(descriptors, CV_32F); 

	// Vector quantization of words in image
	::cv::Mat wordsInImg;
	m_knn->findNearest(descriptors,1,wordsInImg);  // less than 1 ms here
	
	// construction of response histogram
	::std::vector<float> temp(m_vocabulary.rows, 0.0);

	for(int i = 0; i < wordsInImg.rows; ++i)
		temp[wordsInImg.at<float>(i,0)]++;

	// George: Is there a way to avoid transposing?  it is wasted computation, and depending on the implementation can take some time.
	::cv::Mat response_histogram;
	::cv::transpose(::cv::Mat(temp),response_histogram); // make a row-matrix from the column one made from the vector-

	// Normalization of response histogram
	::cv::Mat col;
	for (int i = 0; i < response_histogram.cols; ++i)
	{
		col = response_histogram.col(i);

		col = col - m_scaling_means[i];
		col = col / m_scaling_stds[i];
	}

	response = 0.0;

	try
	{
		response = m_svm->predict(response_histogram, ::cv::noArray(), 0);
		return true;
	}
	catch ( const std::exception & e ) 
	{
		::std::cout << e.what();
		return false;
	}
}

void BagOfFeatures::initializeKNN(::cv::ml::KNearest::Types KNNSearchDataStructure)
{
	m_knn = ::cv::ml::KNearest::create();
	m_knn->setAlgorithmType(KNNSearchDataStructure);

	::cv::Mat mat_words_labels(m_vocabulary.rows, 1, CV_32S);

	for (int i=0;i<m_vocabulary.rows;i++) 
		mat_words_labels.at<int>(i) = i;

	m_knn->clear();
	m_knn->setDefaultK(1);

	m_knn->train(m_vocabulary, ::cv::ml::ROW_SAMPLE, mat_words_labels);

}

void BagOfFeatures::featureExtraction(const ::std::vector<::cv::Mat*>& imgs, ::std::vector<int>& image_number, ::cv::Mat& training_descriptors)
{
	if (training_descriptors.size > 0)
		training_descriptors.resize(0);

	std::vector<cv::KeyPoint> keyPoints;
	::cv::Mat descriptors;

	for(int i = 0; i < imgs.size(); ++i)
	{
		// detect keypoints
		m_featureDetector->detect(*imgs[i], keyPoints);

		// extract descriptors
		m_descriptorExtractor->compute(*imgs[i], keyPoints, descriptors);
		training_descriptors.push_back(descriptors);

		for (int j = 0; j< descriptors.rows; j++)
			image_number.push_back(i);
	}

	// put descriptors in right format for kmeans clustering
	if(training_descriptors.type()!=CV_32F) 
		training_descriptors.convertTo(training_descriptors, CV_32F); 

}

void BagOfFeatures::computeResponseHistogram(const ::std::vector<::cv::Mat*> imgs, const ::cv::Mat& cluster_labels, const ::std::vector<int>& image_number, ::cv::Mat& im_histograms)
{
	::std::vector<float> temp(m_dictionarySize, 0.0);
	::std::vector< ::std::vector<float> > v_im_histograms(imgs.size(), temp);

	for(int i = 0; i < cluster_labels.rows; ++i)
		v_im_histograms[image_number[i]][cluster_labels.at<int>(i)]++;

	im_histograms = vec2cvMat_2D(v_im_histograms);

	m_scaling_means.clear();
	m_scaling_stds.clear();

	// scale the histogram columns
	::cv::Scalar	mean;
	::cv::Scalar	stdev;
	::cv::Mat		col;

	for (int i = 0; i < im_histograms.cols; ++i)
	{

		col = im_histograms.col(i);

		::cv::meanStdDev(col,mean,stdev);

		col = col - mean.val[0];
		col = col / stdev.val[0];

		m_scaling_means.push_back(mean.val[0]);
		m_scaling_stds.push_back(stdev.val[0]);
	}
}