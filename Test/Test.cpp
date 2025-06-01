#include <gtest/gtest.h>
#include "HandDetector.h"
#include <opencv2/opencv.hpp>
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