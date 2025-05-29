#include "HSVMask.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

HSVMask::HSVMask(int hmin_, int hmax_, int smin_, int smax_, int vmin_, int vmax_)
    : hmin(hmin_), hmax(hmax_), smin(smin_), smax(smax_), vmin(vmin_), vmax(vmax_) {

    namedWindow("Trackbars", WINDOW_NORMAL);
    createTrackbar("Hue Min", "Trackbars", &hmin, 179);
    createTrackbar("Hue Max", "Trackbars", &hmax, 179);
    createTrackbar("Sat Min", "Trackbars", &smin, 255);
    createTrackbar("Sat Max", "Trackbars", &smax, 255);
    createTrackbar("Val Min", "Trackbars", &vmin, 255);
    createTrackbar("Val Max", "Trackbars", &vmax, 255);
    createTrackbar("Debug Mode", "Trackbars", &debugModeValue, 1);
}
int HSVMask::getDebugMode() const {
    return debugModeValue;
}
void HSVMask::updateMask(const Mat& src) {
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), mask);
}

Mat HSVMask::getMask() const {
    return mask;
}