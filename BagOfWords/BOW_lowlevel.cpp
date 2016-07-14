#include "BOW_lowlevel.h"
#include "FileUtils.h"


BOW_l::BOW_l()
{

	m_featureDetector = ::cv::xfeatures2d::SURF::create();
	m_descriptorExtractor =  ::cv::xfeatures2d::SURF::create();

	m_dictionarySize = 10000;
	m_tc_Kmeans = ::cv::TermCriteria(::cv::TermCriteria::MAX_ITER + ::cv::TermCriteria::EPS,100000, 0.000001);
	int retries = 1;
	int flags = ::cv::KMEANS_PP_CENTERS;
	m_bowtrainer = new ::cv::BOWKMeansTrainer(50, m_tc_Kmeans, retries, flags);
	
	m_descriptorMatcher = ::cv::DescriptorMatcher::create("FlannBased");
	m_bowide = new ::cv::BOWImgDescriptorExtractor(m_descriptorExtractor, m_descriptorMatcher);
}


BOW_l::~BOW_l()
{

}


/*
Loads the classifier and BOW vocabulary from files
*/
bool BOW_l::LoadFromFile(::std::string path)
{
	try
	{
		m_svm = ::cv::ml::StatModel::load<::cv::ml::SVM>(path + "SVM.xml");

		cv::FileStorage storage(path + "VOC.xml", cv::FileStorage::READ);
		storage["vocabulary"] >> m_vocabulary;
		storage.release();  

		m_bowide->setVocabulary(m_vocabulary);

		return true;
	}
	catch ( const std::exception & e ) 
	{
		::std::cerr << e.what();
		return false;
	}
}


/*
Saves the classifier and BOW vocabulary to files
*/
bool BOW_l::SaveToFile(::std::string path)
{
	try
	{
		m_svm->save(path + "SVM.xml");

		cv::FileStorage storage(path + "VOC.xml", cv::FileStorage::WRITE);
		storage << "vocabulary" << m_vocabulary;
		storage.release();   


		return true;
	}
	catch ( const std::exception & e ) 
	{
		::std::cerr << e.what();
		return false;
	}
}



/*
Trains the BOW with an SVM classifier

argument path is a string pointing to the training dataset. It should contain folders each named after a class, containing positive sample images in the form of PNGs

returns true if the SVM is trained correctly
*/
bool BOW_l::trainBOW(::std::string path)
{

	// declare variables
	std::vector<cv::KeyPoint> keyPoints;
	::cv::Mat descriptors;
	::cv::Mat training_descriptors(0,m_descriptorExtractor->descriptorSize(),m_descriptorExtractor->descriptorType());
	::cv::Mat cluster_labels;

	::cv::Mat img;

	::cv::Mat labels(0, 1, CV_32FC1);
	::cv::Mat trainingData(0, m_dictionarySize, CV_32FC1);
	::cv::Mat response_hist;
	int total_samples = 0;

	// list class names 
	int class_id = 0;
	if (getClassesNames(m_classes, path))
	{
		for each (std::string className in m_classes)
		{
			int count = getImList(m_imList, path + className);
			for (int i=0;i<count;i++) 
			{
				m_classList.push_back(className);
				labels.push_back(class_id);
			}
			class_id++;
		}
	}

	::std::vector<int> image_number;
	// get images and compute keypoints as training descriptors
	for(int i=0; i<m_imList.size();i++)
	{
		std::string filepath = path + m_classList[i] + "\\" + m_imList[i];

		img = ::cv::imread(filepath);
		m_featureDetector->detect(img,keyPoints);
		m_descriptorExtractor->compute(img, keyPoints,descriptors);
		training_descriptors.push_back(descriptors);
		for (int j = 0; j< descriptors.rows; j++) image_number.push_back(i);
	}


	// kmeans classify
	int k = 50;
	::cv::kmeans(training_descriptors, k, cluster_labels, m_tc_Kmeans, 3, cv::KMEANS_PP_CENTERS, m_vocabulary );
	// cluster and set vocabulary into bow ide
	//m_bowtrainer->add(training_descriptors);
	//m_vocabulary = m_bowtrainer->cluster(); 

	::std::vector<float> temp(k, 0.0);
	::std::vector< ::std::vector<float> > v_im_histograms(m_imList.size(),temp);

	for(int i=0; i<cluster_labels.rows;i++)
	{
		int img_number = image_number[i];
		int word = cluster_labels.at<int>(i);

		v_im_histograms[img_number][word] ++;

	}

	::cv::Mat im_histograms = ::cv::Mat::zeros(m_imList.size(),k,CV_32FC1);
	im_histograms = vec2cvMat_2D(v_im_histograms);

	m_scaling_means.clear();
	m_scaling_stds.clear();

	// scale thehistogram columns
	for (int i=0; i<im_histograms.cols;i++)
	{
		::cv::Scalar mean;
		::cv::Scalar stdev;

		::cv::Mat col = im_histograms.col(i);

		::cv::meanStdDev(col,mean,stdev);

		col = col - mean.val[0];
		col = col / stdev.val[0];

		::std::cout << col << ::std::endl;
		::std::cout << " " << ::std::endl;

		m_scaling_means.push_back(mean.val[0]);
		m_scaling_stds.push_back(stdev.val[0]);

	}




	m_bowide->setVocabulary(m_vocabulary);
	//// Setup training data for classifiers
	//for(int i=0; i<m_imList.size();i++)
	//{
	//	std::string filepath = path + m_classList[i] + "\\" + m_imList[i];

	//	img = ::cv::imread(filepath);
	//	m_featureDetector->detect(img,keyPoints);
	//	m_bowide->compute(img, keyPoints, response_hist);

	//	trainingData.push_back(response_hist);

	//	/*int class_id = 0;
	//	for (int j=0;j < m_classes.size();j++)
	//	{
	//		if (m_classes[j] == m_classList[i]) class_id = j;
	//	}
	//	labels.push_back(class_id);*/
	//}  

	m_svm = ::cv::ml::SVM::create();
	//m_svm->setType(::cv::ml::SVM::C_SVC);
	//m_svm->setKernel(::cv::ml::SVM::LINEAR);
	//m_svm->setTermCriteria(::cv::TermCriteria(::cv::TermCriteria::MAX_ITER,1000,0.000001));

	return m_svm->train(im_histograms, ::cv::ml::ROW_SAMPLE,labels);

}

/*
Does a prediction using the BOW

Arguments:
	- path is a string pointing to an image file
	- float is a reference which will take the response

Returns true if there was no runtime error during prediction
*/
bool BOW_l::predictBOW(std::string path, float& response)
{

	std::vector<cv::KeyPoint> keyPoints;

	::cv::Mat img = ::cv::imread(path);
	::cv::Mat bowDescriptor;

	m_featureDetector->detect(img, keyPoints);
	m_bowide->compute(img, keyPoints, bowDescriptor);

	for (int i=0; i<bowDescriptor.cols;i++)
	{
		::cv::Mat col = bowDescriptor.col(i);

		col = col - m_scaling_means[i];
		col = col / m_scaling_stds[i];
	}

	response = 0.0;
	try
	{
		response = m_svm->predict(bowDescriptor);
	}
	catch ( const std::exception & e ) 
	{
		::std::cerr << e.what();
		return false;
	}

	return true;
}
