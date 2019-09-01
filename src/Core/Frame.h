#ifndef FRAME_H
#define FRAME_H

#include "System/Common.h"
#include <DBoW3/Vocabulary.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Frame {
public:
    SMART_POINTER_TYPEDEFS(Frame);

public:
    Frame();

    Frame(const cv::Mat& imBGR, const cv::Mat& imDepth, const double& timeStamp, ExtractorPtr extractor,
       VocabularyPtr voc, CameraPtr cam);

    void extract();

    void computeBoW();

    void setPose(const SE3& Tcw);
    SE3 getPose();
    SE3 getPoseInverse();

    // Check if a Landmark is in the frustum of the camera and fill variables of the Landmark to be used by the tracking
    bool isInFrustum(LandmarkPtr pMP, double viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool posInGrid(const Vec2& kp, int& posX, int& posY);

    std::vector<size_t> getFeaturesInArea(const double& x, const double& y, const double& r, const int minLevel = -1, const int maxLevel = -1) const;

    // Compute 3D keypoints position in camera coordinate system.
    void compute3D();

    void addLandmark(const size_t& i, LandmarkPtr pMP);
    void eraseLandmark(const size_t& i);
    void eraseLandmarks();

    cv::Mat drawMatchedPoints();

    double getDepth(const Vec2& xi);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    VocabularyPtr _vocabulary;

    // Feature extractor
    ExtractorPtr _extractor;

    // Frame timestamp.
    double _timestamp;

    CameraPtr _camera;

    // Number of KeyPoints.
    size_t _N;

    // Features extracted
    std::vector<cv::KeyPoint> _keys;
    cv::Mat _descriptors;
    std::vector<FeaturePtr> _features;

    // Bag of Words Vector structures.
    DBoW3::BowVector _bowVec;
    DBoW3::FeatureVector _featVec;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting Landmarks.
    static double _gridElementWidthInv;
    static double _gridElementHeightInv;
    std::vector<std::size_t> _grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Current and Next Frame id.
    static int _nextId;
    int _id;

    // Reference Keyframe.
    KeyFrame* _refKF;

    // Scale pyramid info.
    int _scaleLevels;
    double _scaleFactor;
    double _logScaleFactor;
    std::vector<double> _scaleFactors;
    std::vector<double> _invScaleFactors;
    std::vector<double> _levelSigma2;
    std::vector<double> _invLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static double _minX;
    static double _maxX;
    static double _minY;
    static double _maxY;

    static bool _initialComputations;

    cv::Mat _colorIm;
    cv::Mat _grayIm;
    cv::Mat _depthIm;

private:
    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void assignFeaturesToGrid();

    // Camera pose.
    SE3 _Tcw;
    SE3 _Twc;
};

#endif // FRAME_H
