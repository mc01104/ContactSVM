#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2/ml.hpp>

#include <iostream> 
#include <stdexcept> 
#include <algorithm>
#include <deque>
#include <chrono>
#include <numeric>

#include "Main.h"
#include "BOW_lowlevel.h"
#include "FileUtils.h"
#include "Network_force.h"
#include "helper_parseopts.h"


/*bool testBOW(std::string path, BOW_l bow)
{

	::std::vector<::std::string> imList;

	int count = getImList(imList,path);

	for(int i=0; i<imList.size();i++)
	{
		float response = 0.0;
		std::string filepath = path + "\\" + imList[i];
		if (bow.predictBOW(filepath,response)) ::std::cout << "Response : " << response << ::std::endl;
		else ::std::cout << "Error in prediction" << ::std::endl;
	}
	return true;
}*/



bool testBOW(std::string path, BOW_l bow, bool visualization = false)
{

	::std::chrono::steady_clock::time_point t1 = ::std::chrono::steady_clock::now();
	::std::chrono::steady_clock::time_point t2 = ::std::chrono::steady_clock::now();

	::std::vector<::std::string> imList;
	int count = getImList(imList,path);
	std::sort(imList.begin(), imList.end(), numeric_string_compare);

	std::deque<float> values(5,0);
	float average_val = 0.0;

	::std::vector<::std::string> classes = bow.getClasses();

	::std::vector<float> reponses;

	int response_contact = 0, response_free = 0;

	std::vector<float> timings;

	for(int i=0; i<imList.size();i++)
	{
		float response = 0.0;
		std::string filepath = path + "\\" + imList[i];

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

			/*float popped = values.front();
			values.pop_front();
			values.push_back(response);

			average_val = (average_val*5.0 - popped + response)/5.0;

			reponses.push_back(response);*/

			if(visualization)
			{
				
				::cv::putText(img,cv::String(::std::to_string(response).c_str()),cv::Point(10,50),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,255,0));
				::cv::imshow("Image", img);
				char key = ::cv::waitKey(1);

				if (key == 27) break;
			}
		}
		else ::std::cout << "Error in prediction" << ::std::endl;
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



void testBOWFeature(std::string feature, std::string train_path, std::string test_path_contact, std::string test_path_free)
{
	BOW_l* bow = new BOW_l(feature);

	if (bow->trainBOW(train_path)) 
	{
		std::cout << "\n\n==============" << std::endl;
		std::cout << feature << " BOW trained and saved" << std::endl;

		::std::cout << "Test with Contact" << ::std::endl;
		testBOW(test_path_contact,*bow, true);

		::std::cout << "Test with Free file" << ::std::endl;
		testBOW(test_path_free,*bow, true);

	}
	delete bow;
}


int main( int argc, char** argv )
{

	std::string base_folder = "M:\\Public\\Data\\Cardioscopy_project\\ContactDetection_data\\Surgery_dev\\";

	std::string train_path = base_folder + "train\\";

	std::string test_path_contact = base_folder + "validate\\Contact\\";
	std::string test_path_free =  base_folder + "validate\\Free\\";

	::std::string test_path_surgery =  base_folder + "..\\..\\2016-05-26_Bypass_Cardioscopy\\Awaiba_Surgery_20160526\\2016-05-26_14-10-11\\";
	test_path_surgery = base_folder + "..\\..\\2016-07-28_Bypass_cardioscopy\\CameraImages_Surgery_07282016\\2016-07-28_12-24-43\\";
	test_path_surgery = base_folder + "..\\..\\2016-07-28_Bypass_cardioscopy\\CameraImages_Surgery_07282016\\2016-07-28_12-10-26\\";
	//test_path_surgery = base_folder + "..\\ExtractedImages_paper\\";

	std::string output_path = base_folder + "output_";

	std::string ip = "192.168.0.12";

	float gain = 3.0;


	if(cmdOptionExists(argv, argv+argc, "-i"))
    {
		char * inputfile = getCmdOption(argv, argv + argc, "-i");
		test_path_surgery  = ::std::string(inputfile);
    }

	if(cmdOptionExists(argv, argv+argc, "-s"))
    {
		char * outputfile = getCmdOption(argv, argv + argc, "-s");
		output_path  = ::std::string(outputfile);
    }

	if(cmdOptionExists(argv, argv+argc, "-ip"))
    {
		char * s_ip = getCmdOption(argv, argv + argc, "-ip");
		ip  = ::std::string(s_ip);
    }

	if(cmdOptionExists(argv, argv+argc, "-g"))
    {
		char * s_gain = getCmdOption(argv, argv + argc, "-g");
		try { gain  = atof(s_gain); }
		catch ( const std::exception & e ) { gain = 3.0;}
    }

	if(cmdOptionExists(argv, argv+argc, "-h"))
    {
		std::cout << "Command line options are: \n -i for input path of image files \n -s for basepath of saved SVM files \n -ip to set the IP address of the server \n -g to set the force gain" << ::std::endl;
		return 0;
    }


	/*testBOWFeature("SURF", train_path, test_path_contact, test_path_free);
	testBOWFeature("FAST-SURF", train_path, test_path_contact, test_path_free);
	testBOWFeature("FAST-LUCID", train_path, test_path_contact, test_path_free);
	*/

	//testBOWFeature("FAST-LUCID", train_path, test_path_contact, test_path_free);

	//system("pause");

	BOW_l bow("FAST-LUCID");

	/*if (bow.trainBOW(train_path))
	{
		//bow.SaveToFile(output_path);

		testBOW(test_path_surgery,bow, false);
	}*/

	if (bow.LoadFromFile(output_path)) 
	{
		testBOW(test_path_contact,bow, true);

		testBOW(test_path_free,bow, true);
	}

	//if (bow.trainBOW(train_path))
	//{
	//	bow.SaveToFile(output_path);

	//	/*::std::cout << "Test with Contact" << ::std::endl;
	//	testBOW(test_path_contact,bow);

	//	::std::cout << "Test with Free file" << ::std::endl;
	//	testBOW(test_path_free,bow);*/
	//}

	//::std::cout << "Load from file test" << ::std::endl;
	//if (bow.LoadFromFile(output_path)) 
	//{
	//	/*::std::cout << "Test with Contact" << ::std::endl;
	//	testBOW(test_path_contact,bow, true);

	//	::std::cout << "Test with Free file" << ::std::endl;
	//	testBOW(test_path_free,bow, true);*/

	//	testBOW(test_path_surgery,bow, true);
	//}
	//else 
	//{
	//	::std::cout << "Error in BOW loading" << ::std::endl;
	//}

	/*Network_force testForce(output_path,test_path_surgery);
	testForce.setForceGain(gain);
	testForce.runThreads();*/

	system("pause");
	return 0;
}