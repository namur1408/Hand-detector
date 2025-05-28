#pragma once
#include "Detector.h"
#include <opencv2/dnn.hpp>
#include <string>

class FaceDetection : public Detector {
private:
    cv::dnn::Net faceNet;

public:
    FaceDetection();
    void detect(cv::Mat& img) override;
    std::string name() const override;
};
