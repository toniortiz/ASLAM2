#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include "System/Common.h"
#include <mutex>
#include <opencv2/features2d.hpp>
#include <string>

class Extractor {
public:
    SMART_POINTER_TYPEDEFS(Extractor);

    enum eAlgorithm {
        ORB = 0,
        ORB2,
        SVO,
        FAST,
        GFTT,
        STAR,
        BRISK,
        FREAK,
        BRIEF,
        LATCH,
        SURF,
        SIFT
    };

    enum eMode {
        NORMAL = 0,
        ADAPTIVE
    };

    eAlgorithm _detectorType, _descriptorType;
    eMode _mode;

    Extractor(eAlgorithm detector, eAlgorithm descriptor, eMode mode);

    void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);

    void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints);
    void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);

    int getLevels();
    double getScaleFactor();
    std::vector<double> getScaleFactors();
    std::vector<double> getInverseScaleFactors();
    std::vector<double> getScaleSigmaSquares();
    std::vector<double> getInverseScaleSigmaSquares();

    inline std::string getDetectorName() { return _detectorName; }
    inline std::string getDescriptorName() { return _descriptorName; }

    void create();

    void print(std::ostream& out, const std::string& msg = "");

    // L2 or HAMMING
    static int _norm;

    // Adaptive params
    int _gridResolution = 3;
    int _iterations = 5;
    int _edgeTh = 31;
    int _minFeats = 600;
    int _maxFeats = 1000;

private:
    void setParameters();

    void init();
    void createAdaptiveDetector();
    void createDetector();
    void createDescriptor();

    cv::Ptr<cv::DescriptorExtractor> _descriptor;
    cv::Ptr<cv::FeatureDetector> _detector;

    std::string _detectorName;
    std::string _descriptorName;

    int _nfeatures;
    double _scaleFactor;
    int _nlevels;
    int _iniThFAST;
    int _minThFAST;

    std::vector<double> _scaleFactors;
    std::vector<double> _invScaleFactors;
    std::vector<double> _levelSigma2;
    std::vector<double> _invLevelSigma2;
};

#endif // EXTRACTOR_H
