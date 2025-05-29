#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "FaceDetection.h"
#include "HSVMask.h"
#include "HandDetector.h"
#include <memory>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Camera not detected!" << endl;
        return -1;
    }

    vector<unique_ptr<Detector>> detectors;
    detectors.push_back(make_unique<FaceDetection>());

    HSVMask hsvMask(0, 179, 0, 123, 94, 255);
    auto handDetector = make_unique<HandDetector>("Resources/fist.xml");
    HandDetector* handPtr = handDetector.get();
    detectors.push_back(std::move(handDetector));

    Mat img, imgMaskSrc;
    while (true) {
        cap.read(img);
        if (img.empty()) continue;

        img.copyTo(imgMaskSrc);
        detectors[0]->detect(imgMaskSrc);

        hsvMask.updateMask(imgMaskSrc);
        handPtr->setMask(hsvMask.getMask());
        handPtr->setDebugMode(hsvMask.getDebugMode());
        detectors[1]->detect(img);
        for (auto& detector : detectors) {
            detector->detect(imgMaskSrc);
            /*cout << detector->name() << " done" << endl;*/
        }
        imshow("Image", img);
        imshow("Mask", hsvMask.getMask());

        if (waitKey(1) == 27) break;
    }
    return 0;
}
