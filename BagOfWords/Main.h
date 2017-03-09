#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include "BOW_lowlevel.h"

bool trainBOW(std::string path, std::string output_path);
bool predictBOW(std::string path, ::cv::Ptr< ::cv::ml::SVM> svm, ::cv::Mat vocabulary);
bool testBOW(std::string path, ::cv::Ptr< ::cv::ml::SVM> svm, ::cv::Mat vocabulary);
bool testBOW(std::string path, BOW_l bow, bool visualization = false, int delay = 1);
bool processFromFile(::std::string csvFilePath, bool trainSVM = true, bool visualize = false, int testType = 0);

void processVideo();
void classifierTest();
