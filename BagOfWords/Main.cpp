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
#include <unistd.h>

#include "Main.h"
#include "BOW_lowlevel.h"
#include "FileUtils.h"
#include "Network_force.h"
#include "helper_parseopts.h"
#include "CSV_reader.h"



void processVideo()
{

    /**** CAREFUL WITH COMPRESSED MP4 VIDEOS !!! *****/

	// Load video
	::std::string filename = "C:\\Users\\RC\\Dropbox\\PVL_robot\\Classifier_test_experiment\\2017-01-26_12-46-26_CR04.mp4";
	::cv::VideoCapture v(filename);
	double num_of_frames = v.get(CV_CAP_PROP_FRAME_COUNT); 

	::std::cout << num_of_frames << ::std::endl;
	//::std::string output_path = "F:\\clparams\\output_";
	//::std::string output_path = "C:\\Users\\RC\Dropbox\\shared_harvard\\classifiers\\class_1\\output_";
	::std::string output_path = "C:\\PASS\\clparams\\output_";
	//::std::string output_for_images = "C:\\Users\\RC\\Dropbox\\Videos\\processed_videos\\clas-cr04\\";
	// Load classifier
	BOW_l bow("FAST-LUCID");
	//BOW_l bow;
	
	bow.LoadFromFile(output_path);
    ::std::vector< ::std::string> classes = bow.getClasses();

	//::std::ofstream contactStream("C:\\Users\\RC\\Dropbox\\Videos\\processed_videos\\ctr_nav_cr_06\\contact_data.txt");

	bool framesLeft = true;
	::cv::Mat frame;
	::cv::namedWindow("MyVideo",CV_WINDOW_AUTOSIZE);

	float response = 0.0;
	::std::vector<int> responses;
	char output_filename[100];
	int counter = 0;

	::std::string final_output;
    for (int i = 0; i < num_of_frames; ++i)
    {
        framesLeft = v.read(frame); 

		if (!bow.predictBOW(frame,response))
			::std::cout << "Classifier failed to estimate contact" << ::std::endl;

		if (classes[(int) response] == "Free")
			response = 0.0;
		else 
			response = 1.0;
		
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

void classifierTest()
{
	BOW_l bow("FAST-LUCID"); //I made a constructor which takes the vocabulary size as a parameter in the linux_compilation branch, but it defaults anyway to 50 in the main branch. You should use BOW_l bow("FAST-LUCID"); in your version of the code, or checkout the linux_compilation branch, but there are hardcoded paths and stuff like this that I did not cleanup

	::std::string train_path = "C:\\Users\\RC\\Documents\\Repos\\software\\ContactSVM\\BagOfWords\\train\\";
	//::std::string output_path = "C:\\Users\\RC\\Documents\\Repos\\software\\ContactSVM\\BagOfWords\\results\\";
	::std::string output_path = "C:\\Users\\RC\\Downloads\\Classifier_test_dataset\\";
	::std::string validate_path_contact = "C:\\Users\\RC\\Documents\\Repos\\software\\ContactSVM\\BagOfWords\\validate\\Contact\\";
	::std::string validate_path_free = "C:\\Users\\RC\\Documents\\Repos\\software\\ContactSVM\\BagOfWords\\validate\\Free\\";

	//if (bow.trainBOW(train_path))
	//{
	//	bow.SaveToFile(output_path);

	//	testBOW(validate_path_contact,bow, false);

	//	testBOW(validate_path_free,bow, false);
	//}

	BOW_l bow2("FAST-LUCID");

	if (bow2.LoadFromFile(output_path))
	{
		testBOW(validate_path_contact,bow2, false);

		testBOW(validate_path_free,bow2, false);

	}
}

bool testBOW(std::string path, BOW_l bow, bool visualization, int delay, bool saveOutput)
{

	::std::chrono::steady_clock::time_point t1 = ::std::chrono::steady_clock::now();
	::std::chrono::steady_clock::time_point t2 = ::std::chrono::steady_clock::now();

    ::std::vector< ::std::string> imList;
	int count = getImList(imList,path);
	std::sort(imList.begin(), imList.end(), numeric_string_compare);

	std::deque<float> values(5,0);
	float average_val = 0.0;

    ::std::vector< ::std::string> classes = bow.getClasses();

	::std::vector<float> reponses;

	int response_contact = 0, response_free = 0;

	std::vector<float> timings;

	for(int i=0; i<imList.size();i++)
	{
		float response = 0.0;
        std::string filepath = path + "/" + imList[i];

		::cv::Mat img = ::cv::imread(filepath);

		t1 = ::std::chrono::steady_clock::now();

		if (bow.predictBOW(img,response)) 
		{

			t2 = ::std::chrono::steady_clock::now();
			std::chrono::microseconds ms = std::chrono::duration_cast<std::chrono::microseconds> (t2-t1);
			timings.push_back(ms.count());

			if (classes[(int) response] == "Free")
			{
				response = 0.0;
				response_free++;
			}
			else 
			{
				response = 1.0;
				response_contact++;
			}

			if(visualization)
			{
				
				::cv::putText(img,cv::String(::std::to_string(response).c_str()),cv::Point(10,50),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,255,0));
				::cv::imshow("Image", img);
                char key = ::cv::waitKey(delay);

				if (key == 27) break;
			}
		}
		else ::std::cout << "Error in prediction" << ::std::endl;
	}

    if (saveOutput)
    {
        // Save all contact values in an XML file
        ::std::string fname = path + "contact_values.xml";
        cv::FileStorage fs(fname, cv::FileStorage::WRITE);
        fs << "contact" << reponses;
        fs.release();
    }

	std::cout << "Number of images: " << imList.size() << ::std::endl;
	std::cout << "Percent of contact: " << response_contact*1.0/imList.size() << ::std::endl;
	std::cout << "Percent of Free: " << response_free*1.0/imList.size() << ::std::endl;
	
	double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
	double mean = sum/1000.0 / timings.size();

	auto result = std::minmax_element(timings.begin(), timings.end());
	
	std::cout << "Average prediction time (ms): " << mean << ::std::endl;

	std::cout << "min is " << *result.first / 1000.0  << ::std::endl;
	std::cout << "max is " << *result.second / 1000.0 << ::std::endl;

	return true;
}


/**
 * @brief: Test the classifier with a list of images (overloaded function using the new BagOfFeatures class)
 *
 * @author: Ben & George
 *
 * \param[in] path - path to the directory containing the test images
 * \param[in] bow - the BagOfFeatures classifier
 * \param[in] visualization - visualize the classified images in an opencv window
 * \param[in] delay - delay during visualization (defaults to 1). O makes it pause at each image waiting keyboard input
 * \param[in] saveOutput - save classifier output in an XML file in the "path" directory
 *
 * Outputs only true for now, should be enhanced to handle errors and exceptions ...
 * */
bool testBOW(std::string path, BagOfFeatures& bow, bool visualization, int delay, bool saveOutput)
{

    ::std::chrono::steady_clock::time_point t1 = ::std::chrono::steady_clock::now();
    ::std::chrono::steady_clock::time_point t2 = ::std::chrono::steady_clock::now();

    ::std::vector< ::std::string> imList;
    int count = getImList(imList,path);
    std::sort(imList.begin(), imList.end(), numeric_string_compare);

    ::std::vector< ::std::string> classes = bow.getClasses();

    ::std::vector<float> reponses;

    int response_contact = 0, response_free = 0;

    std::vector<float> timings;

    for(int i=0; i<imList.size();i++)
    {
        float response = 0.0;
        std::string filepath = path + "/" + imList[i];

        const ::cv::Mat img = ::cv::imread(filepath);

        t1 = ::std::chrono::steady_clock::now();

        if (bow.predict(&img,response))
        {

            t2 = ::std::chrono::steady_clock::now();
            std::chrono::microseconds ms = std::chrono::duration_cast<std::chrono::microseconds> (t2-t1);
            timings.push_back(ms.count());

            if (classes[(int) response] == "Free")
            {
                response = 0.0;
                response_free++;
            }
            else
            {
                response = 1.0;
                response_contact++;
            }

            if(visualization)
            {

                ::cv::putText(img,cv::String(::std::to_string(response).c_str()),cv::Point(10,50),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,255,0));
                ::cv::imshow("Image", img);
                char key = ::cv::waitKey(delay);

                if (key == 27) break;
            }
        }
        else ::std::cout << "Error in prediction" << ::std::endl;
    }

    if (saveOutput)
    {
        // Save all contact values in an XML file
        ::std::string fname = path + "contact_values.xml";
        cv::FileStorage fs(fname, cv::FileStorage::WRITE);
        fs << "contact" << reponses;
        fs.release();
    }

    std::cout << "Number of images: " << imList.size() << ::std::endl;
    std::cout << "Percent of contact: " << response_contact*1.0/imList.size() << ::std::endl;
    std::cout << "Percent of Free: " << response_free*1.0/imList.size() << ::std::endl;

    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum/1000.0 / timings.size();

    auto result = std::minmax_element(timings.begin(), timings.end());

    std::cout << "Average prediction time (ms): " << mean << ::std::endl;

    std::cout << "min is " << *result.first / 1000.0  << ::std::endl;
    std::cout << "max is " << *result.second / 1000.0 << ::std::endl;

    return true;
}


/**
 * @brief: Process images using parameters described in a CSV file
 *
 * The CSV file should contain two fields:
 *      - base_folder: folder where the classifier files and the dataset is (in test/train/validate folders)
 *      - folder_surgeries: base folder with images from surgeries
 *
 * For now, only pre-trained classifier than can be loaded from XML files is implemented
 *
 * @author: Ben & George
 *
 * \param[in] csvFilePath - the path to the CSV file
 * \param[in] trainSVM - train the SVM if true, load from file otherwise. Defaults to true
 * \param[in] visualize - visualize the classifier output in a window or not. Defaults to false
 * \param[in] testType - 0 for surgery images, 1 for validate dataset, 2 for test dataset. Defaults to 0
 *
 * \return true if no error occured
 * */
bool processFromFile(::std::string csvFilePath, bool trainSVM, bool visualize, int testType)
{
    ParseOptions op = ParseOptions(csvFilePath);

    std::string base_folder;
    std::string base_folder_surgeries;

    std::vector<std::string> folder;
    if (op.getData(std::string("base_folder"),folder))
    {
        base_folder = folder[0];
        std::cout << base_folder << std::endl;
    }

    if (op.getData(std::string("folder_surgeries"),folder))
    {
        base_folder_surgeries = folder[0];
        std::cout << base_folder_surgeries << std::endl;
    }

    else
    {
        std::cout << "Problem parsing base folder path from CSV file" << std::endl;
        return 0;
    }

    std::string output_path = base_folder + "output_";
    std::string train_path = base_folder + "/train/";
    std::string validate_path_contact = base_folder + "/validate/Contact/";
    std::string validate_path_free =  base_folder + "/validate/Free/";
    std::string test_path_contact = base_folder + "/test/Contact/";
    std::string test_path_free =  base_folder + "/test/Free/";


    // path of surgery images
    // @TODO: code the path to images directly in the CSV file
    ::std::string test_path_surgery =  base_folder_surgeries + "/2017-01-26_12-42-26/";


    BagOfFeatures bow;


    // TODO: implement function reading the directory to extract the list of images and their labels
    // Equivalent in Bow_lowlevel is the trainBow function

    /*if ((trainSVM)
    {
        ::std::vector< cv::Mat*> imgs;
        getImagesFromPath(imgs,train_path);
        if (!(bow.train(train_path)) )
            return false;
    }*/


    if (! (bow.load(output_path)) )
        return false;

    switch (testType)
    {
        case 1:
            testBOW(validate_path_contact,bow, visualize);
            cv::waitKey(0);
            testBOW(validate_path_free,bow, visualize);
            cv::waitKey(0);
        case 2:
            testBOW(test_path_contact,bow, visualize);
            cv::waitKey(0);
            testBOW(test_path_free,bow, visualize);
            cv::waitKey(0);
        default:
            testBOW(test_path_surgery,bow, visualize);
            cv::waitKey(0);
    }

    return true;
}








int main( int argc, char** argv )
{

    //processVideo();

    ::std::string csvFilePath = "./folders_contactdetection.csv";

    processFromFile(csvFilePath,true,true,0);

	return 0;
}
