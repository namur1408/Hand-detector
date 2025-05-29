#include "HandDetector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

HandDetector::HandDetector(const std::string& cascadePath, bool debug)
    : debugMode(debug) {
    if (!fistCascade.load(cascadePath)) {
        std::cerr << "Error loading cascade: " << cascadePath << std::endl;
        exit(-1);
    }
}

void HandDetector::detect(cv::Mat& output) {
    detectFists(output);
    getContours(mask, output);
}
void HandDetector::setMask(const cv::Mat& maskInput) {
    mask = maskInput;
}
void HandDetector::setDebugMode(bool value) {
    debugMode = value;
}
std::string HandDetector::name() const {
    return "HandDetector";
}

cv::Mat HandDetector::getMask() const {
    return mask;
}

double HandDetector::angleBetween(cv::Point a, cv::Point b, cv::Point c) {
    double ab = cv::norm(a - b);
    double bc = cv::norm(b - c);
    double ac = cv::norm(a - c);
    double angle = acos((ab * ab + bc * bc - ac * ac) / (2 * ab * bc));
    return angle * 180 / CV_PI;
}

void HandDetector::detectFists(const cv::Mat& img) {
    fists.clear();
    fistCascade.detectMultiScale(img, fists, 1.1, 4, 0, cv::Size(50, 50));
    if (debugMode) {
        for (const auto& r : fists) {
            int padX = static_cast<int>(r.width * 0.6);
            int padY = static_cast<int>(r.height * 0.6);
            cv::Rect enlargedR(
                std::max(0, r.x - padX),
                std::max(0, r.y - padY),
                std::min(img.cols - r.x + padX, r.width + 2 * padX),
                std::min(img.rows - r.y + padY, r.height + 2 * padY)
            );
            cv::rectangle(img, enlargedR, cv::Scalar(0, 255, 255), 2);
        }
    }
}

void HandDetector::getContours(cv::Mat imgDil, cv::Mat& drawImg) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgDil, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    bool textDrawn = false;
    for (const auto& contour : contours) {
        int area = cv::contourArea(contour);
        if (area > 8000) {
            if (debugMode)
                cv::drawContours(drawImg, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255, 0, 255), 2);

            std::vector<cv::Point> approxContour;
            cv::approxPolyDP(contour, approxContour, 20, true);
            if (approxContour.size() < 4) continue;

            std::vector<int> hullIndices;
            cv::convexHull(approxContour, hullIndices, false, false);
            if (hullIndices.size() <= 3) continue;

            std::vector<cv::Vec4i> defects;
            try { cv::convexityDefects(approxContour, hullIndices, defects); }
            catch (...) { continue; }

            cv::Rect bbox = cv::boundingRect(approxContour);
            int centerY = bbox.y + bbox.height / 1.5;

            std::vector<cv::Point> fingerTipsCandidates;
            int validDefectsCount = 0;

            for (auto& d : defects) {
                cv::Point start = approxContour[d[0]];
                cv::Point end = approxContour[d[1]];
                cv::Point far = approxContour[d[2]];
                float depth = d[3] / 256.0f;
                double ang = angleBetween(start, far, end);

                if (depth > 55 && ang < 110 && start.y < centerY && end.y < centerY) {
                    fingerTipsCandidates.push_back(start);
                    fingerTipsCandidates.push_back(end);
                    validDefectsCount++;
                }
            }

            std::vector<cv::Point> fingerTips;
            const int thresholdDist = 15;

            if (validDefectsCount >= 1) {
                for (auto& pt : fingerTipsCandidates) {
                    bool tooClose = false;
                    for (auto& added : fingerTips) {
                        if (cv::norm(pt - added) < thresholdDist) {
                            tooClose = true;
                            if (pt.y < added.y) added = pt;
                            break;
                        }
                    }
                    if (!tooClose) {
                        bool insideFist = false;
                        for (const auto& r : fists) {
                            if (r.contains(pt)) {
                                insideFist = true;
                                break;
                            }
                        }
                        if (!insideFist) fingerTips.push_back(pt);
                    }
                }
            }

            if (fingerTips.empty()) {
                cv::Point highest = approxContour[0];
                for (const auto& pt : approxContour) {
                    if (pt.y < highest.y) highest = pt;
                }
                if (highest.y < centerY) {
                    bool insideFist = false;
                    for (const auto& r : fists) {
                        if (r.contains(highest)) {
                            insideFist = true;
                            break;
                        }
                    }
                    if (!insideFist) fingerTips.push_back(highest);
                }
            }

            if (fingerTips.size() > 5) fingerTips.resize(5);

            if (!textDrawn) {
                std::string label = (fingerTips.size() == 0) ? "Fist" : "Fingers: " + std::to_string(fingerTips.size());
                cv::putText(drawImg, label,
                    cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
                textDrawn = true;
            }

            if (debugMode) {
                for (auto& pt : fingerTips) {
                    cv::circle(drawImg, pt, 10, cv::Scalar(0, 255, 0), cv::FILLED);
                }
            }
        }
    }
}