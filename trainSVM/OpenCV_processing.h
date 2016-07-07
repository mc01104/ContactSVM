#pragma once 

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <opencv2/core/core.hpp>


static void convertBGRImageToOpponentColorSpace( const cv::Mat& bgrImage, std::vector<cv::Mat>& opponentChannels )
{
    if( bgrImage.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "input image must be an BGR image of type CV_8UC3" );

    // Prepare opponent color space storage matrices.
    opponentChannels.resize( 3 );
    opponentChannels[0] = cv::Mat(bgrImage.size(), CV_8UC1); // R-G RED-GREEN
    opponentChannels[1] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G-2B YELLOW-BLUE
    opponentChannels[2] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G+B

    for(int y = 0; y < bgrImage.rows; ++y)
        for(int x = 0; x < bgrImage.cols; ++x)
        {
            cv::Vec3b v = bgrImage.at<cv::Vec3b>(y, x);
            uchar& b = v[0];
            uchar& g = v[1];
            uchar& r = v[2];

            opponentChannels[0].at<uchar>(y, x) = cv::saturate_cast<uchar>(0.5f    * (255 + g - r));       // (R - G)/sqrt(2), but converted to the destination data type
            opponentChannels[1].at<uchar>(y, x) = cv::saturate_cast<uchar>(0.25f   * (510 + r + g - 2*b)); // (R + G - 2B)/sqrt(6), but converted to the destination data type
            opponentChannels[2].at<uchar>(y, x) = cv::saturate_cast<uchar>(1.f/3.f * (r + g + b));         // (R + G + B)/sqrt(3), but converted to the destination data type
        }
}