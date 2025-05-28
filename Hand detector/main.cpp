#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace cv;
using namespace dnn;

class Detector {
public:
    virtual void detect(Mat& img) = 0;
    virtual string name() const = 0;
    virtual ~Detector() {}
};
class FaceDetection : public Detector {
private:
    String faceModelFile = "Resources/res10_300x300_ssd_iter_140000.caffemodel";
    String faceConfigFile = "Resources/deploy.prototxt.txt";
    Net faceNet = readNetFromCaffe(faceConfigFile, faceModelFile);
public:
    FaceDetection() {
        if (faceNet.empty()) {
            cout << "Could not load face DNN model." << endl;
            exit(-1);
        }
    }
    void detect(Mat& img) override {
        Mat blobFace = blobFromImage(img, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
        faceNet.setInput(blobFace);
        Mat faceDetections = faceNet.forward();
        Mat faceMat(faceDetections.size[2], faceDetections.size[3], CV_32F, faceDetections.ptr<float>());
        float confidenceThreshold = 0.65;

        for (int i = 0; i < faceMat.rows; i++) {
            float confidence = faceMat.at<float>(i, 2);
            if (confidence > confidenceThreshold) {
                int x1 = static_cast<int>(faceMat.at<float>(i, 3) * img.cols);
                int y1 = static_cast<int>(faceMat.at<float>(i, 4) * img.rows);
                int x2 = static_cast<int>(faceMat.at<float>(i, 5) * img.cols);
                int y2 = static_cast<int>(faceMat.at<float>(i, 6) * img.rows);
                int rectHeight = y2 - y1;
                y2 = min(img.rows, y2 + static_cast<int>(rectHeight * 0.2));
                Rect faceRect(x1, y1, x2 - x1, y2 - y1);
                rectangle(img, faceRect, Scalar(0, 0, 0), FILLED);
            }
        }
    }
    string name() const override {
        return "FaceDetection";
    }
};

class HSVMask {
private:
    int hmin, hmax, smin, smax, vmin, vmax;
    Mat mask;

public:
    HSVMask(int hmin_, int hmax_, int smin_, int smax_, int vmin_, int vmax_)
        : hmin(hmin_), hmax(hmax_), smin(smin_), smax(smax_), vmin(vmin_), vmax(vmax_) {
        namedWindow("Trackbars", WINDOW_AUTOSIZE);
        createTrackbar("Hue Min", "Trackbars", &hmin, 179);
        createTrackbar("Hue Max", "Trackbars", &hmax, 179);
        createTrackbar("Sat Min", "Trackbars", &smin, 255);
        createTrackbar("Sat Max", "Trackbars", &smax, 255);
        createTrackbar("Val Min", "Trackbars", &vmin, 255);
        createTrackbar("Val Max", "Trackbars", &vmax, 255);
    }

    void updateMask(const Mat& src) {
        Mat hsv;
        cvtColor(src, hsv, COLOR_BGR2HSV);
        Scalar lower(hmin, smin, vmin);
        Scalar upper(hmax, smax, vmax);
        inRange(hsv, lower, upper, mask);
    }

    Mat getMask() const {
        return mask;
    }
};

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Camera not detected!\n";
        return -1;
    }
    vector<unique_ptr<Detector>> detectors;
    detectors.push_back(make_unique<FaceDetection>());
    HSVMask hsvMask(0, 179, 0, 123, 94, 255);

    Mat img, imgMaskSrc;
    while (true) {
        cap.read(img);
        if (img.empty()) continue;

        img.copyTo(imgMaskSrc);
        detectors[0]->detect(imgMaskSrc);
        cout << detectors[0]->name() << " done" << endl;

        hsvMask.updateMask(imgMaskSrc);
        imshow("Image", img);
        imshow("Mask", hsvMask.getMask());

        if (waitKey(1) == 27) break;
    }
    return 0;
}