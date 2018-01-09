#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include "classifier.h"
#include "dataset.h"
#include "LineDetection.h"

bool testBOW(std::string path, BagOfFeatures& bow, bool visualization = false, int delay = 50, bool saveOutput = false);
bool testLineDetection(std::string path, LineDetector& lDetector, bool visualization = false, int delay = 140, bool saveOutput = false);

bool processFromFile(::std::string csvFilePath, bool trainSVM = true, bool visualize = false);

void createDataset(const ::std::string& path, ::std::vector< ::cv::Mat>& images, ::std::vector<int>& labels);

void trainClassifier(const ::std::string& train_path);
bool trainClassifier(const ::std::string& train_path, BagOfFeatures& bow);


void classifierTestGeorge();
void processVideoWithClassifier(const ::std::string& video_path, const ::std::string& video_filename, const BagOfFeatures& bow);
void processImagesWithClassifier(const ::std::string& images_path, const BagOfFeatures& bow);

bool processFromFileLineDetection(::std::string csvFilePath, bool visualize = false);