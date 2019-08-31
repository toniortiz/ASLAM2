#include "DetectorAdjuster.h"
#include <iostream>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

DetectorAdjuster::DetectorAdjuster(const Extractor::eAlgorithm& detector, double initialThresh, double minThresh,
    double maxThresh, double increaseFactor, double decreaseFactor)
    : _thresh(initialThresh)
    , _minThresh(minThresh)
    , _maxThresh(maxThresh)
    , _increaseFactor(increaseFactor)
    , _decreaseFactor(decreaseFactor)
    , _detectorAlgorithm(detector)
{
    if (!(detector == Extractor::eAlgorithm::SURF || detector == Extractor::eAlgorithm::SIFT
            || detector == Extractor::eAlgorithm::FAST || detector == Extractor::eAlgorithm::GFTT
            || detector == Extractor::eAlgorithm::ORB)) {
        cerr << "Not supported detector" << endl;
    }
}

void DetectorAdjuster::detect(cv::InputArray image, vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    cv::Ptr<cv::Feature2D> detector;

    if (_detectorAlgorithm == Extractor::eAlgorithm::FAST)
        detector = cv::FastFeatureDetector::create(_thresh);
    else if (_detectorAlgorithm == Extractor::eAlgorithm::GFTT)
        detector = cv::GFTTDetector::create(10000, 0.01, _thresh, 3, false, 0.04);
    else if (_detectorAlgorithm == Extractor::eAlgorithm::ORB)
        detector = cv::ORB::create(10000, 1.2f, 8, 15, 0, 2, 0, 31, static_cast<int>(_thresh));
    else if (_detectorAlgorithm == Extractor::eAlgorithm::SURF)
        detector = cv::xfeatures2d::SurfFeatureDetector::create(_thresh);
    else if (_detectorAlgorithm == Extractor::eAlgorithm::SIFT)
        detector = cv::xfeatures2d::SiftFeatureDetector::create(0, 3, _thresh);

    detector->detect(image, keypoints);
}

void DetectorAdjuster::setDecreaseFactor(double new_factor) { _decreaseFactor = new_factor; }

void DetectorAdjuster::setIncreaseFactor(double new_factor) { _increaseFactor = new_factor; }

void DetectorAdjuster::tooFew(int, int)
{
    _thresh *= _decreaseFactor;
    if (_thresh < _minThresh)
        _thresh = _minThresh;
}

void DetectorAdjuster::tooMany(int, int)
{
    _thresh *= _increaseFactor;
    if (_thresh > _maxThresh)
        _thresh = _maxThresh;
}

bool DetectorAdjuster::good() const
{
    return (_thresh > _minThresh) && (_thresh < _maxThresh);
}

cv::Ptr<DetectorAdjuster> DetectorAdjuster::clone() const
{
    cv::Ptr<DetectorAdjuster> pNewObject(new DetectorAdjuster(_detectorAlgorithm, _thresh, _minThresh, _maxThresh, _increaseFactor, _decreaseFactor));
    return pNewObject;
}
