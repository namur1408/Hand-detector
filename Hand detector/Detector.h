#pragma once
#include <opencv2/core.hpp>
#include <string>
class Detector {
public:
    virtual void detect(cv::Mat& img) = 0;
    virtual std::string name() const = 0;
    virtual ~Detector() {}
};
