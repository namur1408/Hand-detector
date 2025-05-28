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
        cout << "Camera not detected!" << endl;;
        return -1;
    }

    vector<unique_ptr<Detector>> detectors;
    detectors.push_back(make_unique<FaceDetection>());
    HSVMask hsvMask(0, 179, 0, 123, 94, 255);
    HandDetector hand("Resources/fist.xml");
    Mat img, imgMaskSrc;
    while (true) {
        cap.read(img);
        if (img.empty()) continue;
        img.copyTo(imgMaskSrc);
        detectors[0]->detect(imgMaskSrc);
        cout << detectors[0]->name() << " done" << endl;
        hsvMask.updateMask(imgMaskSrc);
        hand.detect(hsvMask.getMask(), img);
        imshow("Image", img);
        imshow("Mask", hsvMask.getMask());

        if (waitKey(1) == 27) break;
    }
    return 0;
}