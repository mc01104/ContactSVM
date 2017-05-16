#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include "classifier.h"
#include "dataset.h"


bool testBOW(std::string path, BagOfFeatures& bow, bool visualization = false, int delay = 40, bool saveOutput = false, bool writeVideo = false);

bool testBOW_hierarchical(std::string path, BagOfFeatures& bow, bool visualization = false, int delay = 30, bool saveOutput = false);


bool processFromFile(::std::string csvFilePath, bool trainSVM = true, bool visualize = false);

void createDataset(const ::std::string& path, ::std::vector< ::cv::Mat>& images, ::std::vector<int>& labels);

void trainClassifier(const ::std::string& train_path);
bool trainClassifier(const ::std::string& train_path, BagOfFeatures& bow);


void classifierTestGeorge();
void processVideoWithClassifier(const ::std::string& video_path, const ::std::string& video_filename, BagOfFeatures& bow);
void processImagesWithClassifier(const ::std::string& images_path, BagOfFeatures& bow);

