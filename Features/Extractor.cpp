#include "Extractor.h"
#include "DetectorAdjuster.h"
#include "ORBextractor.h"
#include "SVOextractor.h"
#include "StatefulFeatureDetector.h"
#include "VideoDynamicAdaptedFeatureDetector.h"
#include "VideoGridAdaptedFeatureDetector.h"
#include <iostream>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

int Extractor::_norm;

Extractor::Extractor(eAlgorithm detector, eAlgorithm descriptor, eMode mode)
{
    _detectorType = detector;
    _descriptorType = descriptor;
    _mode = mode;

    setParameters();
    create();
}

void Extractor::setParameters()
{
    if (_detectorType == ORB || _detectorType == ORB2) {
        _nfeatures = 1000;
        _scaleFactor = 1.2;
        _nlevels = 8;
        _iniThFAST = 20;
        _minThFAST = 7;
    } else if (_detectorType == FAST) {
        _nfeatures = 1000;
        _scaleFactor = 1.0;
        _nlevels = 1;
        _iniThFAST = 10; // By default is 10
    } else if (_detectorType == SVO) {
        _nfeatures = 1000;
        _scaleFactor = 1.5;
        _nlevels = 3;
        _iniThFAST = 20;
    } else if (_detectorType == GFTT) {
        _nfeatures = 1300;
        _scaleFactor = 1.0;
        _nlevels = 1;
    } else if (_detectorType == STAR) {
        _nfeatures = 1000;
        _scaleFactor = 1.0;
        _nlevels = 1;
    } else if (_detectorType == BRISK) {
        _nfeatures = 1000;
        _scaleFactor = 1.2;
        _nlevels = 8;
        _iniThFAST = 20;
    } else if (_detectorType == SURF) {
        _nfeatures = 1000;
        _scaleFactor = 1.0;
        _nlevels = 4;
    } else if (_detectorType == SIFT) {
        _nfeatures = 1000;
        _scaleFactor = 1.0;
        _nlevels = 3;
    }
}

void Extractor::create()
{
    if (_mode == ADAPTIVE) {
        if (_detectorType == FAST || _detectorType == GFTT || _detectorType == SURF
            || _detectorType == SIFT || _detectorType == ORB) {
            createAdaptiveDetector();
            init();
        } else {
            cerr << "Not supported adaptive detector" << endl;
            terminate();
        }
    } else if (_mode == NORMAL) {
        createDetector();
        init();
    }

    createDescriptor();
    _norm = _descriptor->defaultNorm();
}

void Extractor::createDetector()
{
    switch (_detectorType) {
    case ORB:
        _detector = cv::ORB::create(_nfeatures, _scaleFactor, _nlevels);
        _detectorName = "ORB";
        break;

    case FAST:
        _detector = cv::FastFeatureDetector::create(_iniThFAST, true, cv::FastFeatureDetector::TYPE_9_16);
        _detectorName = "FAST";
        break;

    case SVO:
        _detector.reset(new SVOextractor(_nlevels, _gridResolution, _iniThFAST));
        _detectorName = "SVO";
        break;

    case GFTT:
        _detector = cv::GFTTDetector::create(_nfeatures, 0.01, 3, 3, false, 0.04);
        _detectorName = "GFTT";
        break;

    case STAR:
        _detector = cv::xfeatures2d::StarDetector::create();
        _detectorName = "STAR";
        break;

    case BRISK:
        _detector = cv::BRISK::create(_iniThFAST, _nlevels, _scaleFactor);
        _detectorName = "BRISK";
        break;

    case SURF:
        _detector = cv::xfeatures2d::SURF::create(100, _nlevels, 3, false, false);
        _detectorName = "SURF";
        break;

    case SIFT:
        _detector = cv::xfeatures2d::SIFT::create(_nfeatures, _nlevels);
        _detectorName = "SIFT";
        break;

    case ORB2:
        _detector.reset(new ORBextractor(_nfeatures, _scaleFactor, _nlevels, _iniThFAST, _minThFAST));
        _detectorName = "ORB2";
        break;

    default:
        cout << "Invalid detector!" << endl;
        std::terminate();
    }
}

void Extractor::createDescriptor()
{
    switch (_descriptorType) {
    case ORB:
        _descriptor = cv::ORB::create();
        _descriptorName = "ORB";
        break;

    case BRISK:
        _descriptor = cv::BRISK::create();
        _descriptorName = "BRISK";
        break;

    case BRIEF:
        _descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        _descriptorName = "BRIEF";
        break;

    case FREAK:
        _descriptor = cv::xfeatures2d::FREAK::create();
        _descriptorName = "FREAK";
        break;

    case SURF:
        _descriptor = cv::xfeatures2d::SURF::create();
        _descriptorName = "SURF";
        break;

    case SIFT:
        _descriptor = cv::xfeatures2d::SIFT::create();
        _descriptorName = "SIFT";
        break;

    case LATCH:
        _descriptor = cv::xfeatures2d::LATCH::create();
        _descriptorName = "LATCH";
        break;

    case ORB2:
        // Only for update mNorm
        _descriptor = cv::ORB::create();
        _descriptorName = "ORB2";
        break;

    default:
        cout << "Invalid descriptor!" << endl;
        std::terminate();
    }
}

void Extractor::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    if (_detectorType == ORB2 && _descriptorType == ORB2) {
        _detector->detectAndCompute(image, mask, keypoints, descriptors);
    } else {
        _detector->detect(image, keypoints);
        if (keypoints.size() > _nfeatures)
            cv::KeyPointsFilter::retainBest(keypoints, _nfeatures);

        _descriptor->compute(image, keypoints, descriptors);
    }
}

void Extractor::detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints)
{
    _detector->detect(image, keypoints);

    if (keypoints.size() > _nfeatures)
        cv::KeyPointsFilter::retainBest(keypoints, _nfeatures);
}

void Extractor::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    _descriptor->compute(image, keypoints, descriptors);
}

void Extractor::init()
{
    _scaleFactors.clear();
    _levelSigma2.clear();
    _invScaleFactors.clear();
    _invLevelSigma2.clear();

    _scaleFactors.resize(_nlevels);
    _levelSigma2.resize(_nlevels);
    _scaleFactors[0] = 1.0;
    _levelSigma2[0] = 1.0;

    if (_nlevels > 1) {
        for (int i = 1; i < _nlevels; i++) {
            _scaleFactors[i] = _scaleFactors[i - 1] * _scaleFactor;
            _levelSigma2[i] = _scaleFactors[i] * _scaleFactors[i];
        }
    }

    _invScaleFactors.resize(_nlevels);
    _invLevelSigma2.resize(_nlevels);
    for (int i = 0; i < _nlevels; i++) {
        _invScaleFactors[i] = 1.0 / _scaleFactors[i];
        _invLevelSigma2[i] = 1.0 / _levelSigma2[i];
    }
}

void Extractor::createAdaptiveDetector()
{
    DetectorAdjuster* pAdjuster = nullptr;

    if (_detectorType == FAST) {
        pAdjuster = new DetectorAdjuster(_detectorType, 20, 2, 10000, 1.3, 0.7);
        _detectorName = "FAST";
    } else if (_detectorType == GFTT) {
        pAdjuster = new DetectorAdjuster(_detectorType, 3, 1, 10, 1.3, 0.7);
        _detectorName = "GFTT";
    } else if (_detectorType == SURF) {
        pAdjuster = new DetectorAdjuster(_detectorType, 200, 2, 10000, 1.3, 0.7);
        _detectorName = "SURF";
    } else if (_detectorType == SIFT) {
        pAdjuster = new DetectorAdjuster(_detectorType, 0.04, 0.0001, 10000, 1.3, 0.7);
        _detectorName = "SIFT";
    } else if (_detectorType == ORB) {
        pAdjuster = new DetectorAdjuster(_detectorType, 20, 2, 10000, 1.3, 0.7);
        _detectorName = "ORB";
    }

    int gridCells = _gridResolution * _gridResolution;
    int gridMin = round(_minFeats / static_cast<float>(gridCells));
    int gridMax = round(_maxFeats / static_cast<float>(gridCells));

    StatefulFeatureDetector* pDetector = new VideoDynamicAdaptedFeatureDetector(pAdjuster, gridMin, gridMax, _iterations);
    _detector.reset(new VideoGridAdaptedFeatureDetector(pDetector, _maxFeats, _gridResolution, _gridResolution, _edgeTh));
}

int Extractor::getLevels()
{
    return _nlevels;
}

double Extractor::getScaleFactor()
{
    return _scaleFactor;
}

vector<double> Extractor::getScaleFactors()
{
    return _scaleFactors;
}

vector<double> Extractor::getInverseScaleFactors()
{
    return _invScaleFactors;
}

vector<double> Extractor::getScaleSigmaSquares()
{
    return _levelSigma2;
}

vector<double> Extractor::getInverseScaleSigmaSquares()
{
    return _invLevelSigma2;
}

void Extractor::print(ostream& out, const string& msg)
{
    if (msg.size() > 0)
        out << msg << endl;
    out << (_mode == NORMAL ? "Normal" : "Adaptive") << " feature extraction" << endl;
    out << " -Detector: " << _detectorName << "\n -Descriptor: " << _descriptorName << endl;
}
