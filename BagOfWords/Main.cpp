#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include <iostream> 
#include <stdexcept> 
#include <algorithm>
#include <deque>
#include <chrono>
#include <numeric>

#include <limits.h>

#ifdef LINUX
	#include <unistd.h>
#endif

#include "Main.h"
#include "FileUtils.h"
#include "helper_parseopts.h"
#include "CSV_reader.h"

#include "LineDetection.h"
#include "FilterLibrary.h"

int find_opencv_version();

/**
 * @brief: Test the classifier with a list of images or a video
 *
 * @author: Ben & George
 *
 * \param[in] path - path to the video or to the directory containing the test images
 * \param[in] bow - the BagOfFeatures classifier
 * \param[in] visualization - visualize the classified images in an opencv window
 * \param[in] delay - delay during visualization (defaults to 1). O makes it pause at each image waiting keyboard input
 * \param[in] saveOutput - save classifier output in an XML file in the "path" directory
 *
 * Outputs only true for now, should be enhanced to handle errors and exceptions ...
 * */
bool testBOW(std::string path, BagOfFeatures& bow, bool visualization, int delay, bool saveOutput)
{

    // Variables declaration and initialization
    ::std::vector< ::std::string> imList;
    int count = 0;

    ::std::vector<float> reponses;
    int response_contact = 0, response_free = 0; //??
	int response_tissue = 0;
	int response_chordae = 0;
    ::std::vector< ::std::string> classes = bow.getClasses();

    std::vector<float> timings;
    ::std::chrono::steady_clock::time_point t1 = ::std::chrono::steady_clock::now();
    ::std::chrono::steady_clock::time_point t2 = ::std::chrono::steady_clock::now();

	
    // Open a videocapture if the testpath is a video file
    bool isVideo = false;
    ::cv::VideoCapture cap;

    //::std::vector< ::std::string> vid_extensions = {'.avi', '.mp4'};     // this line doesn't work for me somehow... I tried other compilers as well. The syntax is correct but I cannot compile it with the initializer list
	::std::vector<::std::string> vid_extensions;
	vid_extensions.push_back(".avi");
	vid_extensions.push_back(".mp4");
	
    for ( ::std::string ext : vid_extensions)
    {
        if (path.find(ext)!=std::string::npos)
        {
            isVideo = true;
            cap.open(path);
        }
    }

    if (!isVideo)
    {
        count = getImList(imList, checkPath(path + "/" ));
        std::sort(imList.begin(), imList.end(), numeric_string_compare);
    }

    int img_index = 0;

    while (img_index < count)
    {
        float response = 0.0;
        ::cv::Mat img;

        if (isVideo)
        {
            if (!cap.read(img))
                break;
        }
        else
        {
            if (img_index > imList.size())
                break;
            std::string filepath = checkPath(path + "/" + imList[img_index]);
            img = ::cv::imread(filepath);
        }

        t1 = ::std::chrono::steady_clock::now();

        if (bow.predict(img,response))
        {
   //         t2 = ::std::chrono::steady_clock::now();
   //         std::chrono::microseconds ms = std::chrono::duration_cast<std::chrono::microseconds> (t2-t1);
   //         timings.push_back(ms.count());

   //         if (classes[(int) response] == "Tissue")
   //         {
   //             response = 0.0;
   //             response_tissue++;
   //         }
   //         else if (classes[(int) response] == "Free")
   //         {
   //             response = 1.0;
   //             response_free++;
   //         }
			//else if (classes[(int) response] == "Chordae")
			//{
			//	response = 2.0;
			//	response_chordae++;
			//}
			//else if (classes[(int) response] == "Valve")
			//{
			//	response = 3.0;
			//	response_contact++;
			//}

            if(visualization)
            {

                //::cv::putText(img,cv::String(::std::to_string(response).c_str()),cv::Point(10,50),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,255,0));
				::cv::putText(img,cv::String(classes[(int) response]),cv::Point(10,50),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,255,0));
                ::cv::imshow("Image", img);
                char key = ::cv::waitKey(delay);

                if (key == 27) break;
            }
        }
        else ::std::cout << "Error in BOW prediction" << ::std::endl;


        img_index ++;
    }

    if (saveOutput)
    {
        // Save all contact values in an XML file
        ::std::string fname = checkPath(path + "/" + "contact_values.xml");
        cv::FileStorage fs(fname, cv::FileStorage::WRITE);
        fs << "contact" << reponses;
        fs.release();
    }

 //   // not necessarily all images are processed, so imList.size() is not appropriate
 //   // timings has a number of elements equal to the number of processed images
 //   std::cout << "Number of images: " << timings.size() << ::std::endl;
 //   std::cout << "Percent of Valve: " << response_contact*1.0/timings.size() << ::std::endl;
 //   std::cout << "Percent of Free: " << response_free*1.0/timings.size() << ::std::endl;
	//std::cout << "Percent of Tissue: " << response_tissue*1.0/timings.size() << ::std::endl;

 //   double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
 //   double mean = sum/1000.0 / timings.size();

 //   auto result = std::minmax_element(timings.begin(), timings.end());

 //   std::cout << "Average prediction time (ms): " << mean << ::std::endl;

 //   std::cout << "min is " << *result.first / 1000.0  << ::std::endl;
 //   std::cout << "max is " << *result.second / 1000.0 << ::std::endl;

    return true;
}


/**
 * @brief: Process images using parameters described in a CSV file
 *
 * The CSV file should contain three fields:
 *      - train_path: folder where the training dataset is (should have subdirectories with classes names containing png images)
 *      - output_path: base path for saving/loading the classifier files
 *      - test_path: path with test images
 *
 * Additionnaly, it has two optional fields:
 *      - trainSVM: 1 for training the SVM, 0 to load it from the saved files
 *      - visualize: 1 for visualizing test results in a window
 *
 * @author: Ben & George
 *
 * \param[in] csvFilePath - the path to the CSV file
 * \param[in] trainSVM - train the SVM if true, load from file otherwise. Defaults to true, overrided if present in CSV file
 * \param[in] visualize - visualize the classifier output in a window or not. Defaults to false, overrided if present in CSV file
 *
 * \return true if no error occured
 * */
bool processFromFile(::std::string csvFilePath, bool trainSVM, bool visualize)
{
    ParseOptions op = ParseOptions(csvFilePath);

    ::std::string train_path, output_path,test_path;
    ::std::vector< ::std::string> folder;
    ::std::vector< float> numData;

	bool labelFlag = false;

    bool pathOK = true;

    if (op.getData(std::string("train_path"),folder))
    {
        train_path = folder[0];
        std::cout << "Train path: " << train_path << std::endl;
    }
    else pathOK = false;

    if (op.getData(std::string("output_path"),folder))
    {
        output_path = folder[0];
        std::cout << "Ouput path: " << output_path << std::endl;
    }
    else pathOK = false;

    if (op.getData(std::string("test_path"),folder))
    {
        test_path = folder[0];
        std::cout << "Test path: " << test_path << std::endl;
    }
    else pathOK = false;

    // One of the mandatory options (paths) is missing from the file
    if (!pathOK)
    {
        std::cout << "Problem parsing base folder path from CSV file" << std::endl;
        return 0; // TODO: raise exception ???
    }


    // optional stuff
    if (op.getData(std::string("trainSVM"),numData))
    {
        trainSVM = numData[0];
    }

    if (op.getData(std::string("visualize"),numData))
    {
        visualize = numData[0];
    }

    if (op.getData(std::string("label"),numData))
    {
        labelFlag = numData[0];
    }


    BagOfFeatures bow; 

    if (trainSVM)
    {
        // the dataset object is only needed when training a classifier
        Dataset dataset;
        dataset.initDataset(train_path);
        if (!(bow.train(dataset)))
            return false; // TODO: raise exception ???

        // saving by default, an option could be added later on
        bow.save(output_path);
        dataset.serializeInfo(output_path);
    }
	else if (labelFlag)
	{
        Dataset dataset;
        dataset.initUnlabelledDataset(test_path);
		bow.load(output_path);
		bow.autoLabelImages(dataset);
	}
    else
    {
        if (! (bow.load(output_path)) )
            return false; // TODO: raise exception ???
    }

    testBOW(test_path,bow, visualize);
    cv::waitKey(0);

    return true;
}


/**
 * @brief: Create a dataset from a given path
 *
 * @author: Ben & George
 *
 * \param[in] path - path to the directory containing the dataset. Should contain "Contact" and "Free" subdirectories with images
 * \param[in,out] images - a std::vector of openCV Mat images
 * \param[in,out] labels - the class labels corresponding to the images
 * */
void createDataset(const ::std::string& path, ::std::vector< ::cv::Mat>& images, ::std::vector<int>& labels)
{

	// load the training images
    ::std::string tempPath = path + "Contact/";
    ::std::vector< ::std::string> im_paths_train_contact;

	// load contact and create labels
	int count_train_contact = getImList(im_paths_train_contact, tempPath);
	for (int i = 0; i < count_train_contact; ++i)
		im_paths_train_contact[i] = tempPath + im_paths_train_contact[i];

	::std::vector<int> labels_contact(count_train_contact);
	for (int i = 0; i < count_train_contact; ++i)
		labels_contact[i] = 1;

	// load free and create labels
    tempPath = path + "Free/";
    ::std::vector< ::std::string> im_paths_train_free;
	int count_train_free = getImList(im_paths_train_free, tempPath);
	for (int i = 0; i < count_train_free; ++i)
		im_paths_train_free[i] = tempPath + im_paths_train_free[i];

	::std::vector<int> labels_free(count_train_free);
	for (int i = 0; i < count_train_free; ++i)
		labels_free[i] = 0;


	// build the training data
    ::std::vector< ::std::string> imgPaths(count_train_contact);
	::std::copy(im_paths_train_contact.begin(), im_paths_train_contact.end(), imgPaths.begin());
	imgPaths.insert(imgPaths.end(), im_paths_train_free.begin(), im_paths_train_free.end());

	labels.resize(count_train_contact);
	::std::copy(labels_contact.begin(), labels_contact.end(), labels.begin());
	labels.insert(labels.end(), labels_free.begin(), labels_free.end());

	if (imgPaths.size() != labels.size())
		throw("Error creating the training dataset -> images and labels are not equal in number");

	for (int i = 0; i < imgPaths.size(); ++i)
		images.push_back(::cv::imread(imgPaths[i]));

}


/*====================================
 * == DEPRECATED FUNCTIONS, CLEAN ? ==
 * ==================================*/


/**
 * @brief: train a bow classifier and save it to the current working directory [deprecated]
 *
 * @author: Ben & George
 *
 * \param[in] path - path to the directory containing the dataset. Should contain "Contact" and "Free" subdirectories with images
 * */
void trainClassifier(const ::std::string& train_path)
{
	// need to give as input the number of words
    ::std::vector< ::cv::Mat> training_imgs;
	::std::vector<int> training_labels;
	//::std::string train_path = "C:\\Users\\RC\\Documents\\Repos\\software\\ContactSVM\\BagOfWords\\train\\";
	createDataset(train_path, training_imgs, training_labels);
	
	BagOfFeatures bow;
	bow.train(training_imgs, training_labels);
	bow.save("./");
}


/**
 * @brief: train a bow classifier [deprecated]
 *
 * @author: Ben & George
 *
 * \param[in] path - path to the directory containing the dataset. Should contain "Contact" and "Free" subdirectories with images
 * \param[in,out] bow - the BagOfFeatures classifier object
 * */
bool trainClassifier(const ::std::string& train_path, BagOfFeatures& bow)
{
    // need to give as input the number of words
    ::std::vector< ::cv::Mat> training_imgs;
    ::std::vector<int> training_labels;

    createDataset(train_path, training_imgs, training_labels);

    // @FIXME: stupid hack to get the classes into the bow
    Dataset dataset;
    dataset.initDataset(train_path);
    bow.setClasses(dataset.getClasses());

    return bow.train(training_imgs, training_labels);
}



/*================================
 * == LEGACY FUNCTIONS, CLEAN ? ==
 * =============================*/

void classifierTestGeorge()
{
	//::std::string train_path = "C:\\Users\\RC\\Documents\\Repos\\software\\ContactSVM\\BagOfWords\\train\\";
	//::std::string train_path = "C:\\Users\\RC\\Dropbox\\Classifier_test_dataset\\train\\";

	//trainClassifier(train_path);

	BagOfFeatures bow;
	bow.load(".\\classifier_surgery\\");

	//::std::string video_path = "Z:\\Public\\Data\\Cardioscopy_project\\2017-01-26_bypass_cardioscopy\\Videos_2017-01-26\\";
	//::std::string video_filename = "2017-01-26_12-42-26_classifier test_setup.avi";
	//processVideoWithClassifier(video_path, video_filename, bow);


	::std::string image_paths = "Z:\\Public\\Data\\Cardioscopy_project\\2017-01-26_bypass_cardioscopy\\Videos_2017-01-26\\2017-01-26_15-00-57\\";
	processImagesWithClassifier(image_paths, bow);

}


void processImagesWithClassifier(const ::std::string& images_path, const BagOfFeatures& bow)
{

    ::std::vector< ::std::string> imgPaths;
	double num_of_frames = getImList(imgPaths, images_path); //conventionally the input arguments go first followed by the output arguments (code styling comment)

	std::sort(imgPaths.begin(), imgPaths.end(), numeric_string_compare);

	::std::cout << num_of_frames << ::std::endl;

	
	::std::string output_path = "Z:\\Public\\Data\\Cardioscopy_project\\2017-01-26_bypass_cardioscopy\\class_post_process\\2017-01-26_15-00-57\\";
	
	::std::ofstream contactStream(output_path + "contact_data.txt");
	
	::cv::Mat frame;
	::cv::namedWindow("MyVideo",CV_WINDOW_AUTOSIZE);

	float response = 0.0;
	::std::vector<int> responses;
	char output_filename[100];
	int counter = 0;
	int framesLeft;
	::std::string final_output;
    for (int i = 0; i < num_of_frames; ++i)
    {
		frame = ::cv::imread(images_path + imgPaths[i]);

		if (!bow.predict(frame, response))
			::std::cout << "Classifier failed to estimate contact" << ::std::endl;
	
		responses.push_back(response);

		::cv::Point center(20,50);
		::cv::Scalar color(0,255,255);

		if (response == 1)
			::cv::circle(frame, center, 10, color, -1);

		sprintf(output_filename,"image_%010d.png", ++counter);
		final_output = output_path + output_filename;

		//::std::cout << output_filename << ::std::endl;
		::cv::imwrite(final_output, frame);
		contactStream << response << ::std::endl;
		//::std::cout << "response:" << response << ::std::endl;
        //imshow("MyVideo", frame);          
        //if(::cv::waitKey(0) == 27) break;
    }

	contactStream.close();

}


void processVideoWithClassifier(const ::std::string& video_path, const ::std::string& video_filename, const BagOfFeatures& bow)
{
	::cv::VideoCapture v(video_path + video_filename);
	double num_of_frames = v.get(CV_CAP_PROP_FRAME_COUNT); 

	::std::cout << num_of_frames << ::std::endl;

	//::std::ofstream contactStream("C:\\Users\\RC\\Dropbox\\Videos\\processed_videos\\ctr_nav_cr_06\\contact_data.txt");


	::cv::Mat frame;
	::cv::namedWindow("MyVideo",CV_WINDOW_AUTOSIZE);

	float response = 0.0;
	::std::vector<int> responses;
	char output_filename[100];
	int counter = 0;
	int framesLeft;
	::std::string final_output;
    for (int i = 0; i < num_of_frames; ++i)
    {
		framesLeft = v.read(frame); 

		if (!bow.predict(frame, response))
			::std::cout << "Classifier failed to estimate contact" << ::std::endl;
	
		responses.push_back(response);

		::cv::Point center(20,50);
		::cv::Scalar color(0,255,255);

		if (response == 1)
			::cv::circle(frame, center, 10, color, -1);

		//sprintf(output_filename,"image_%010d.png", ++counter);
		//final_output = output_for_images + output_filename;

		//::std::cout << output_filename << ::std::endl;
		//::cv::imwrite(final_output, frame);
		//contactStream << response << ::std::endl;
		::std::cout << "response:" << response << ::std::endl;
        imshow("MyVideo", frame);          
        if(::cv::waitKey(0) == 27) break;
    }

	//contactStream.close();

}




/*====================
 * == MAIN FUNCTION ==
 * ==================*/

int main( int argc, char** argv )
{

	::std::string csvFilePath = "./folders_contactdetection_example_g.csv";
    processFromFile(csvFilePath);
	//::std::string csvFilePath = "./folders_linedetection_example_g.csv";
	//processFromFileLineDetection(csvFilePath);
	return 0;
}

 
int find_opencv_version()
{
  ::std::cout << "OpenCV version : " << CV_VERSION << endl;
  ::std::cout << "Major version : " << CV_MAJOR_VERSION << endl;
  ::std::cout << "Minor version : " << CV_MINOR_VERSION << endl;
  ::std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;

  return 0;
 }


bool processFromFileLineDetection(::std::string csvFilePath, bool visualize)
{
    ParseOptions op = ParseOptions(csvFilePath);

    ::std::string test_path;
    ::std::vector< ::std::string> folder;
    ::std::vector< float> numData;

    bool pathOK = true;

    if (op.getData(std::string("test_path"),folder))
    {
        test_path = folder[0];
        std::cout << "Test path: " << test_path << std::endl;
    }
    else pathOK = false;

    // One of the mandatory options (paths) is missing from the file
    if (!pathOK)
    {
        std::cout << "Problem parsing base folder path from CSV file" << std::endl;
        return 0; // TODO: raise exception ???
    }


    if (op.getData(std::string("visualize"),numData))
    {
        visualize = numData[0];
    }

    LineDetector lDetector;

	testLineDetection(test_path, lDetector, visualize);
    cv::waitKey(0);

    return true;

}


bool testLineDetection(std::string path, LineDetector& lDetector, bool visualization, int delay, bool saveOutput)
{

	// video
    ::cv::VideoWriter writer;
    int codec = CV_FOURCC('M', 'P', 'E', 'G');  // select desired codec (must be available at runtime)
    double fps = 25.0;                          // framerate of the created video stream
    string filename = "./line_1.avi";             // name of the output video file
    writer.open(filename, codec, fps, ::cv::Size(250, 250));
    // check if we succeeded
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

	// Variables declaration and initialization
	::std::vector< ::std::string> imList;
	int count = getImList(imList, checkPath(path + "/" ));
	std::sort(imList.begin(), imList.end(), numeric_string_compare);


	int img_index = 0;
	
	cv::Vec4f line;
	cv::Vec2f centroid;

	RecursiveFilter::RecursiveMovingAverage m_radius_filter;
	RecursiveFilter::RecursiveMovingAverage m_theta_filter;
	m_theta_filter.setDistance(&angularDistanceMinusPItoPI);

	bool lineDetected = false;

	::std::ofstream os("debug.txt");

	while (img_index < count)
	{
		::cv::Mat img;

		if (img_index > imList.size())
			break;

		std::string filepath = checkPath(path + "/" + imList[img_index]);
		img = ::cv::imread(filepath);
        
		double r, theta;
		lineDetected = lDetector.processImage(img, line, centroid);
		if (lineDetected)
		{
			// adjust for the cropping
			::Eigen::Vector2d centroidEig;
			centroidEig(0) = centroid[0];
			centroidEig(1) = centroid[1];

			::Eigen::Vector2d tangentEig;
			tangentEig[0] = line[0];
			tangentEig[1] = line[1];
			tangentEig.normalize();

			::Eigen::Vector2d image_center((int) img.rows/2, (int) img.rows/2);

			// bring to polar coordinated to perform filtering and then move back to point + tangent representation

			::Eigen::VectorXd closest_point;
			nearestPointToLine(image_center, centroidEig, tangentEig, closest_point);
			cartesian2DPointToPolar(closest_point.segment(0, 2) - image_center, r, theta);

			//os << r << " " << theta;

			// filter
			::std::cout << img_index << ", radius:" << r << ", theta:" << theta * 180/M_PI;
			r = m_radius_filter.step(r);
			theta = m_theta_filter.step(theta);

			::std::cout << ", radius:" << r << ", theta:" << theta * 180/M_PI << ::std::endl;

			//os << " " << r << " " << theta * 180/M_PI << " " << theta << ::std::endl;

			//bring back to centroid-tangent
			centroidEig(0) = r * cos(theta);
			centroidEig(1) = r * sin(theta);

			::Eigen::Vector2d filtered_tangent;
			computePerpendicularVector(centroidEig, tangentEig);

			centroidEig += image_center;


			// -----------------------------//

			// find closest point from center to line -> we will bring that point to the center of the images
			double lambda = (image_center - centroidEig).transpose() * tangentEig;
			centroidEig += lambda * tangentEig;

			::cv::line( img, ::cv::Point(centroidEig(0), centroidEig(1)), ::cv::Point(centroidEig(0)+tangentEig(0)*100, centroidEig(1)+tangentEig(1)*100), ::cv::Scalar(0, 255, 0), 2, CV_AA);
			::cv::line( img, ::cv::Point(centroidEig(0), centroidEig(1)), ::cv::Point(centroidEig(0)+tangentEig(0)*(-100), centroidEig(1)+tangentEig(1)*(-100)), ::cv::Scalar(0, 255, 0), 2, CV_AA);
			::cv::circle(img, ::cv::Point(centroidEig[0], centroidEig[1]), 5, ::cv::Scalar(255,0,0));

		}
		::cv::imshow("test", img);
		::cv::waitKey(10);
		writer.write(img);
		img_index++;




	}
	os.close();
	writer.release();
	return true;
}