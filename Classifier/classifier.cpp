#include "classifier.h"

#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
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
ImageClassifier::predict(const ::std::vector<::cv::Mat>& imgs, ::std::vector<float>& labels) 
{
	if (imgs.size() < 1)
        // std::exception(char*) is a MS-specific function, not in the C++ standards
        throw(::std::runtime_error("Vector of images is empty!!"));

	float tmpResponse = 0.0;
	labels.resize(0);

	for (::std::vector<::cv::Mat>::const_iterator it = imgs.begin(); it != imgs.end(); ++it)
	{
		if(!this->predict(*it, tmpResponse))
			return false;

		labels.push_back(tmpResponse);
	}

	return true;
}

//@TODO - supports only LUCID feature descirptors at the moment
BagOfFeatures::BagOfFeatures(bool gridFeatures):
	ImageClassifier(),
	m_dictionarySize(0),
	m_trained(false),
    m_flannKNN(true),
	m_gridFeatures(gridFeatures)
{
	m_featureDetector =  cv::FastFeatureDetector::create();
    m_descriptorExtractor = cv::xfeatures2d::LUCID::create(5,1);
    //m_descriptorExtractor = cv::xfeatures2d::SURF::create(10,2,2,false,true);

	m_tc_Kmeans = ::cv::TermCriteria(::cv::TermCriteria::MAX_ITER + ::cv::TermCriteria::EPS,100000, 0.000001);

}

BagOfFeatures::~BagOfFeatures()
{
}


/**
 * @brief: Load a BagOfFeatures classifier from a set of XML files
 *
 * The XML files are :
 *      VOC.xml - the BOF vocabulary
 *      KNN.xml - the trained k-nearest-neighbor (optional)
 *      SVM.xml - the trained SVM parameters and decision functions
 *      SCALE_[means,std].xml - the scaling parameters
 *      CLASSES.xml - names of the classes in the classifier
 *
 * @author: Ben & George
 *
 * \param[in] path_to_classifier_files - the path to the directory containing the classifier XMLs
 *
 * \return true if the BOF classifier was correctly loaded and no error occured
 * */
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
            if (m_flannKNN)
                m_flann_index.load(m_vocabulary,path_to_classifier_files + "KNN.xml");

			::cv::FileStorage readKNN(path_to_classifier_files +"KNN.xml", ::cv::FileStorage::READ);
			m_knn = ::cv::Algorithm::read<::cv::ml::KNearest>(readKNN.root());

			m_trained = true;	 // not sure if I need that
		}
		else
		{
			m_dictionarySize = 50;
			//::cv::kmeans(training_descriptors, m_dictionarySize, cluster_labels, m_tc_Kmeans, 3, cv::KMEANS_PP_CENTERS, m_vocabulary );
		}

		// initialize k-nearest neighbors
        this->initializeKNN(::cv::ml::KNearest::BRUTE_FORCE, m_flannKNN);
		
		// Load SVM parameters
		m_svm = ::cv::ml::StatModel::load<::cv::ml::SVM>(path_to_classifier_files + "SVM.xml");

        // Get dictionary size from loaded vocabulary #rows
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


/**
 * @brief: Save a BagOfFeatures classifier to a set of XML files
 *
 * The XML files are the same as in the BagOfFeatures::load function
 *
 * @author: Ben & George
 *
 * \param[in] path_to_classifier_files - the path to the directory containing the classifier XMLs
 *
 * \return true if the BOF classifier was correctly saved and no error occured
 * */
bool BagOfFeatures::save(const ::std::string& path_to_classifier_files)
{
	try
	{
        if (!m_flannKNN)
            m_knn->save(path_to_classifier_files + "KNN.xml");
        else
            m_flann_index.save(path_to_classifier_files + "KNN.xml");
		
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

/**
 * @brief: Train a BagOfFeatures classifier
 *
 * @author: Ben & George
 *
 * \param[in] imgs - a vector of cv::Mat* elements containing the training images
 * \param[in] labels - a vector containing the class labels associated with the training images
 *
 * \return true if the BOF classifier was correctly trained
 * */
bool BagOfFeatures::train(const ::std::vector<::cv::Mat>& imgs, const ::std::vector<int>& labels)
{
	
    ::std::cout << "Extracting features ...." << ::std::endl;

	// feature extraction
	::std::vector<int> image_number; // this looks as if it could be done in a more elegant way
	::cv::Mat training_descriptors(0, m_descriptorExtractor->descriptorSize(), m_descriptorExtractor->descriptorType());
	featureExtraction(imgs, image_number, training_descriptors, m_gridFeatures);

    ::std::cout << "Extracted " << training_descriptors.rows << " features" << ::std::endl;

    ::std::cout << "Vocabulary construction ...." << ::std::endl;

	// kmeans cluster to construct the vocabulary
	::cv::Mat cluster_labels;
    m_dictionarySize = 500;

    m_flannKNN = false;

    if (m_flannKNN)
    {
        ::cv::Mat voc_temp = cv::Mat::zeros(m_dictionarySize, training_descriptors.cols, CV_32F);

        ::cvflann::KMeansIndexParams kmean_params(32, 10, ::cvflann::FLANN_CENTERS_KMEANSPP);
        m_dictionarySize = ::cv::flann::hierarchicalClustering< ::cv::flann::L2<float> >(training_descriptors,voc_temp,kmean_params);
        ::std::cout << m_dictionarySize << ::std::endl;

        voc_temp(::cv::Range(0,m_dictionarySize), ::cv::Range(0, voc_temp.cols)).copyTo(m_vocabulary);

    }
    else
        ::cv::kmeans(training_descriptors, m_dictionarySize, cluster_labels, m_tc_Kmeans, 3, cv::KMEANS_PP_CENTERS, m_vocabulary );

    ::std::cout << "KNN initialization ...." << ::std::endl;

    this->initializeKNN(::cv::ml::KNearest::BRUTE_FORCE, m_flannKNN);

    if (m_flannKNN)
    {
        ::cv::Mat dists;
        m_flann_index.knnSearch(training_descriptors,cluster_labels,dists,1);
    }
    m_flannKNN = true;


    ::std::cout << "Computing response histogram ...." << ::std::endl;

	// compute response histograms for all training images
	::cv::Mat im_histograms = ::cv::Mat::zeros(imgs.size(), m_dictionarySize, CV_32FC1);
	computeResponseHistogram(imgs, cluster_labels, image_number, im_histograms);

    ::std::cout << "Training SVM" << ::std::endl;

	// train SVM classifier
	m_svm = ::cv::ml::SVM::create();
    m_svm->setGamma(1.0f/m_dictionarySize);

	/*::cv::Mat labelsCV(labels, CV_32FC1);*/
	//::std::vector<int> labelsInt(labels.size());
	//::std::copy(labels.begin(), labels.end(), labelsInt.begin());
	m_trained = (m_svm->train(im_histograms, ::cv::ml::ROW_SAMPLE, labels) ? true : false); 


    ::std::cout << "BoF trained" << ::std::endl;


	return m_trained;

}



/**
 * @brief: Train a BagOfFeatures classifier - overloaded function to use a dataset class as input
 *
 * @author: Ben & George
 *
 * \param[in] dataset - a dataset
 *
 * \return true if the BOF classifier was correctly trained
 * */
bool BagOfFeatures::train(const Dataset& dataset)
{
    if (!dataset.isInit())
        return false;

    setClasses(dataset.getClasses());
    return train(dataset.getImages(),dataset.getLabels());
}



/**
 * @brief: Predict the class of a given input image
 *
 * @author: Ben & George
 *
 * \param[in] img - a cv::Mat* input image
 * \param[in,out] response - the class predicted by the BOF classifier
 *
 * \return true if the BOF classifier was correctly trained
 * */
bool BagOfFeatures::predict(const ::cv::Mat img, float& response) 
{
	if (!m_trained) 
		return false;

	std::vector<cv::KeyPoint> keyPoints;
	::cv::Mat descriptors;

	std::vector<int> v_word_labels;
	for (int i = 0; i < m_vocabulary.rows; ++i) 
		v_word_labels.push_back(i);

	if(!m_gridFeatures)
		m_featureDetector->detect(img, keyPoints);
	else
		this->gridKeypointExtraction(img, keyPoints);


	m_descriptorExtractor->compute(img, keyPoints,descriptors);
	
	// put descriptors in right format for kmeans clustering
	if(descriptors.type()!=CV_32F) 
		descriptors.convertTo(descriptors, CV_32F); 

	// Vector quantization of words in image
	::cv::Mat wordsInImg;

    if (m_flannKNN)
    {
        ::cv::Mat dists;
        m_flann_index.knnSearch(descriptors,wordsInImg,dists,1);
    }
    else
        m_knn->findNearest(descriptors,1,wordsInImg);
	
	// construction of response histogram
	::std::vector<float> temp(m_vocabulary.rows, 0.0);

	for(int i = 0; i < wordsInImg.rows; ++i)
		temp[wordsInImg.at<float>(i,0)]++;

	// George: Is there a way to avoid transposing?  it is wasted computation, and depending on the implementation can take some time.
	::cv::Mat response_histogram;
	::cv::transpose(::cv::Mat(temp),response_histogram); // make a row-matrix from the column one made from the vector-

	// Normalization of response histogram

//	response_histogram /= ::cv::norm(response_histogram, ::cv::NormTypes::NORM_L2, ::cv::Mat());


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


/**
 * @brief: Accessor method to get the list of classes in the BOF classifier
 *
 * @author: Ben & George
 *
 * \return a vector of strings containing the classes
 * */
::std::vector< ::std::string> BagOfFeatures::getClasses()
{
    return m_classes;
}


/**
 * @brief: Mutator method to set the list of classes in the BOF classifier
 *
 * @author: Ben & George
 *
 * \return a vector of strings containing the classes
 * */
void BagOfFeatures::setClasses(::std::vector< ::std::string> classes)
{
    m_classes = classes;
}




/**
 * @brief: Initialize the K-Nearest-Neighbor
 *
 * @author: Ben & George
 *
 * \param[in] KNNSearchDataStructure - the type of KNN. 1 is BruteForce and 2 is KDTree. KDTree implementation is currently broken
 *
 * */
void BagOfFeatures::initializeKNN(::cv::ml::KNearest::Types KNNSearchDataStructure, bool flann)
{

    if (flann)
        initializeKNN_flann();

    else
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
}




/**
 * @brief: Initialize the K-Nearest-Neighbor using the FLANN implementation
 *
 * @author: Ben & George
 *
 *
 * */
void BagOfFeatures::initializeKNN_flann()
{
    ::cv::flann::AutotunedIndexParams flann_params(0.7,0.3,0,0.5);
    //m_flann_index = ::cv::flann::Index(m_vocabulary, flann_params);
    m_flann_index.build(m_vocabulary,flann_params);//,::cvflann::FLANN_DIST_HAMMING);
}






/**
 * @brief: Extract image features
 *
 * @author: Ben & George
 *
 * \param[in] imgs - a vector of openCV images
 * \param[in,out] image_number - a vector making the correspondence between the descriptors and the input image they belong to
 * \param[in,out] training_descriptors - a cv::Mat with all descriptors from all images stacked vertically
 * */
void BagOfFeatures::featureExtraction(const ::std::vector<::cv::Mat>& imgs, ::std::vector<int>& image_number, ::cv::Mat& training_descriptors, bool gridFeatures)
{
	if (training_descriptors.size > 0)
		training_descriptors.resize(0);

	std::vector<cv::KeyPoint> keyPoints;
	::cv::Mat descriptors;

	for(int i = 0; i < imgs.size(); ++i)
	{
		// detect keypoints
		if (!gridFeatures)
			m_featureDetector->detect(imgs[i], keyPoints);
		else
			this->gridKeypointExtraction(imgs[i], keyPoints);


		// extract descriptors
		m_descriptorExtractor->compute(imgs[i], keyPoints, descriptors);
		training_descriptors.push_back(descriptors);

		for (int j = 0; j< descriptors.rows; j++)
			image_number.push_back(i);
	}

	// put descriptors in right format for kmeans clustering
	if(training_descriptors.type()!=CV_32F) 
		training_descriptors.convertTo(training_descriptors, CV_32F); 

}


/**
 * @brief: Compute the response histogram for given input images
 *
 * @author: Ben & George
 *
 * \param[in] imgs - a vector of openCV images
 * \param[in] cluster_labels - labels of the words in vocabulary, clustered by the KNN algorithm
 * \param[in] image_number - a vector making the correspondence between the cluster_labels and the input image they belong to
 * \param[in,out] im_histograms - scaled response histograms for each image in imgs
 * */
void BagOfFeatures::computeResponseHistogram(const ::std::vector<::cv::Mat>& imgs, const ::cv::Mat& cluster_labels, const ::std::vector<int>& image_number, ::cv::Mat& im_histograms)
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
	::cv::Mat		row;

//	for (int i = 0; i < im_histograms.rows; ++i)
//	{
//		row = im_histograms.row(i);
//		row /= ::cv::norm(row, ::cv::NormTypes::NORM_L2, ::cv::Mat());
//	}

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


void BagOfFeatures::autoLabelImages(const Dataset& dataset)
{
	::cv::Mat feature_voc;
	//::std::vector<::cv::Mat> hists;
	::cv::Mat hists;
	::std::vector<int> labels;
	this->computeClustersInFeatureSpace(dataset, feature_voc, hists);

	this->applyClusteringInFeatureSpace(dataset, feature_voc, hists, labels);

	this->saveImages(dataset, labels);
}

void BagOfFeatures::saveImages(const Dataset& dataset, const ::std::vector<int>& labels)
{
	::std::vector<::std::string> img_paths = dataset.getImagesList();
	::std::vector<::cv::Mat> imgs = dataset.getImages();
	::std::string mainPath = dataset.getMainPath();
	for (int i = 0; i < labels.size(); ++i)
		::cv::imwrite(checkPath(mainPath + "/" + num2str(labels[i]) + "/" + img_paths[i]), imgs[i]);
}

void BagOfFeatures::applyClusteringInFeatureSpace(const Dataset& dataset, const ::cv::Mat& feature_voc, ::cv::Mat& resp, ::std::vector<int>& labels)
{
	//::std::vector<int> clusterLabel;

	// kmeans and knn needs to become a class...
	::cv::Ptr< ::cv::ml::KNearest> knn;
	knn = ::cv::ml::KNearest::create();
    knn->setAlgorithmType(::cv::ml::KNearest::Types::BRUTE_FORCE);

	::cv::Mat mat_words_labels(feature_voc.rows, 1, CV_32S);

	for (int i = 0; i < feature_voc.rows; ++i) 
		mat_words_labels.at<int>(i) = i;

	knn->clear();
	knn->setDefaultK(1);

	knn->train(feature_voc, ::cv::ml::ROW_SAMPLE, mat_words_labels);	

	for (int i = 0; i < resp.rows; ++i)
		labels.push_back(knn->predict(resp.row(i)));

}

void BagOfFeatures::computeClustersInFeatureSpace(const Dataset& dataset, ::cv::Mat& feature_voc, ::cv::Mat& resp)
{
	std::vector<cv::KeyPoint> keyPoints;
	::cv::Mat descriptors;

	std::vector<int> v_word_labels;
	for (int i = 0; i < m_vocabulary.rows; ++i) 
		v_word_labels.push_back(i);

	::std::vector<::cv::Mat> imgs = dataset.getImages();
	::std::vector<::cv::Mat> response_hists;

	//::cv::Mat resp;
	for (int i = 0; i < imgs.size(); ++i)
	{
		m_featureDetector->detect(imgs[i], keyPoints);
		m_descriptorExtractor->compute(imgs[i], keyPoints,descriptors);
	
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

		resp.push_back(response_histogram);
	}

	::cv::Mat cluster_labels_training;
	::cv::TermCriteria tc_Kmeans = ::cv::TermCriteria(::cv::TermCriteria::MAX_ITER + ::cv::TermCriteria::EPS,100000, 0.000001);
	::cv::kmeans(resp, 3, cluster_labels_training, tc_Kmeans, 3, cv::KMEANS_PP_CENTERS, feature_voc);
}


void 
BagOfFeatures::gridKeypointExtraction(const ::cv::Mat img, ::std::vector<::cv::KeyPoint>& keypoints, int step)
{
	if (keypoints.size() > 0)
		keypoints.clear();

	for (int y = step; y < img.rows - step; y += step)
		for (int x = step; x < img.cols - step; x += step)
			keypoints.push_back(::cv::KeyPoint(float(x), float(y), float(step)));

}
