#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>
#include "HSVMask.h"

class HandDetector {
public:
    HandDetector(const std::string& cascadePath, bool debug = true);
    void detect(const cv::Mat& srcMaskImg, cv::Mat& output);
    std::string name() const;
    cv::Mat getMask() const;

private:
    double angleBetween(cv::Point a, cv::Point b, cv::Point c);
    void detectFists(const cv::Mat& img);
    void getContours(cv::Mat mask, cv::Mat& drawImg);

    std::vector<cv::Rect> fists;
    cv::CascadeClassifier fistCascade;
    cv::Mat mask;
    bool debugMode;
};