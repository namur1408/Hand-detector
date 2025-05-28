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
int main() {

	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "Camera not detected!\n";
		return -1;
	}
	vector<unique_ptr<Detector>> detectors;
	detectors.push_back(make_unique<FaceDetection>());
	Mat img;
	while (true) {
		cap.read(img);
		if (img.empty()) continue;
		for (auto& detector : detectors) {
			detector->detect(img);
			cout << detector->name() << " done" << endl;
		}
		imshow("Image", img);
		waitKey(1);
	}
	return 0;
}