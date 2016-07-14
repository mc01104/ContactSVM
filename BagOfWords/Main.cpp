#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2/ml.hpp>

#include <iostream> 
#include <stdexcept> 
#include <algorithm>

#include "Main.h"
#include "BOW_lowlevel.h"
#include "FileUtils.h"


bool testBOW(std::string path, BOW_l bow)
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
}





int main( int argc, char** argv )
{

	/*
	if(cmdOptionExists(argv, argv+argc, "-d"))
    {
		// load dictionary from file
    }

    char * inputfile = getCmdOption(argv, argv + argc, "-i");

	char * outputfile = getCmdOption(argv, argv + argc, "-o");
	*/

	std::string path = "C:\\Users\\CH182482\\Documents\\Software\\Python_scripts\\bag-of-words\\contactdetect\\train\\";
	std::string test_path_contact = "C:\\Users\\CH182482\\Documents\\Software\\Python_scripts\\bag-of-words\\contactdetect\\test\\Contact\\";
	std::string test_path_free = "C:\\Users\\CH182482\\Documents\\Software\\Python_scripts\\bag-of-words\\contactdetect\\test\\Free\\";
	std::string output_path = "C:\\Users\\CH182482\\Documents\\Software\\Python_scripts\\bag-of-words\\contactdetect\\output_";

	BOW_l bow;

	if (bow.trainBOW(path))
	{
		bow.SaveToFile(output_path);

		::std::cout << "Test with Contact" << ::std::endl;
		testBOW(test_path_contact,bow);

		::std::cout << "Test with Free file" << ::std::endl;
		testBOW(test_path_free,bow);
	}

	/*::std::cout << "Load from file and re-test" << ::std::endl;
	if (bow.LoadFromFile(output_path)) testBOW(test_path,bow);

	else 
	{
		::std::cout << "Error in BOW loading" << ::std::endl;
	}*/

	system("pause");
	return 0;
}