#include <gtest/gtest.h>
#include "HandDetector.h"
#include "FaceDetection.h"
#include "HSVMask.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
namespace fs = std::filesystem;
const std::string kCascadePath = "../../Test/Resources/fist.xml";
class HandDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        detector = std::make_unique<HandDetector>(kCascadePath, false);
    }
    std::unique_ptr<HandDetector> detector;
};

TEST_F(HandDetectorTest, CascadeLoadedSuccessfully) {
 
    ASSERT_NO_THROW({
        HandDetector det(kCascadePath);
        });
}
TEST_F(HandDetectorTest, DetectDoesNotThrow) {
    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::Mat dummyMask = cv::Mat::ones(480, 640, CV_8UC1) * 255;
    detector->setMask(dummyMask);
    ASSERT_NO_THROW({
        detector->detect(img);
        });
    cv::Mat mask = detector->getMask();
    EXPECT_FALSE(mask.empty());
}
TEST_F(HandDetectorTest, DebugModeToggle) {
    detector->setDebugMode(true);
    EXPECT_TRUE(detector->getDebugMode());
    detector->setDebugMode(false);
    EXPECT_FALSE(detector->getDebugMode());
}
