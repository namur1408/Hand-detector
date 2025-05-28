#pragma once
#include <opencv2/core.hpp>

class HSVMask {
private:
    int hmin, hmax, smin, smax, vmin, vmax;
    cv::Mat mask;

public:
    HSVMask(int hmin, int hmax, int smin, int smax, int vmin, int vmax);
    void updateMask(const cv::Mat& src);
    cv::Mat getMask() const;
};
