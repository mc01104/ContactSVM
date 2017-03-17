#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include "BOW_lowlevel.h"
#include "classifier.h"
#include "dataset.h"

bool trainBOW(std::string path, std::string output_path);
bool predictBOW(std::string path, ::cv::Ptr< ::cv::ml::SVM> svm, ::cv::Mat vocabulary);
bool testBOW(std::string path, ::cv::Ptr< ::cv::ml::SVM> svm, ::cv::Mat vocabulary);
bool testBOW(std::string path, BOW_l bow, bool visualization = false, int delay = 1, bool saveOutput = false);
bool testBOW(std::string path, BagOfFeatures& bow, bool visualization = false, int delay = 1, bool saveOutput = false);



bool processFromFile(::std::string csvFilePath, bool trainSVM = true, bool visualize = false);

void processVideo();
void classifierTest();

void classifierTestGeorge();
void processVideoWithClassifier(const ::std::string& video_path, const ::std::string& video_filename, const BagOfFeatures& bow);
void processImagesWithClassifier(const ::std::string& images_path, const BagOfFeatures& bow);


void createDataset(const ::std::string& path, ::std::vector< ::cv::Mat>& images, ::std::vector<int>& labels);
bool trainClassifier(::std::string& train_path, BagOfFeatures& bow);
void trainClassifier(const ::std::string& train_path);
bool trainClassifier(const Dataset& dataset, BagOfFeatures& bow);
