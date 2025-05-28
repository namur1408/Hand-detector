#include "FaceDetection.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;

FaceDetection::FaceDetection() {
    String model = "Resources/res10_300x300_ssd_iter_140000.caffemodel";
    String config = "Resources/deploy.prototxt.txt";
    faceNet = readNetFromCaffe(config, model);
    if (faceNet.empty()) {
        std::cerr << "Could not load face DNN model.\n";
        exit(-1);
    }
}

void FaceDetection::detect(Mat& img) {
    Mat blob = blobFromImage(img, 1.0, Size(300, 300), Scalar(104, 177, 123));
    faceNet.setInput(blob);
    Mat detections = faceNet.forward();
    Mat detMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

    for (int i = 0; i < detMat.rows; i++) {
        float confidence = detMat.at<float>(i, 2);
        if (confidence > 0.65) {
            int x1 = static_cast<int>(detMat.at<float>(i, 3) * img.cols);
            int y1 = static_cast<int>(detMat.at<float>(i, 4) * img.rows);
            int x2 = static_cast<int>(detMat.at<float>(i, 5) * img.cols);
            int y2 = static_cast<int>(detMat.at<float>(i, 6) * img.rows);
            y2 = std::min(img.rows, y2 + static_cast<int>((y2 - y1) * 0.2));
            rectangle(img, Rect(x1, y1, x2 - x1, y2 - y1), Scalar(0, 0, 0), FILLED);
        }
    }
}

std::string FaceDetection::name() const {
    return "FaceDetection";
}