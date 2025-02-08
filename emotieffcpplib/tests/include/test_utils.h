#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

std::string getEmotiEffLibRootDir();
std::vector<cv::Mat> recognizeFaces(const cv::Mat& frame, int downscaleWidth = 500);
std::string getPathToPythonTestDir();

#endif
