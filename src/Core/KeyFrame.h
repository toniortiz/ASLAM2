#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "System/Common.h"
#include <DBoW3/BowVector.h>
#include <DBoW3/FeatureVector.h>
#include <DBoW3/Vocabulary.h>
#include <mutex>
#include <octomap/octomap.h>
#include <opencv2/core.hpp>
#include <set>

class KeyFrame {
public:
    SMART_POINTER_TYPEDEFS(KeyFrame);

    KeyFrame(Frame& F, MapPtr pMap, KeyFrameDatabasePtr pKFDB);

    // Pose functions
    void setPose(const SE3& Tcw);
    SE3 getPose();
    SE3 getPoseInverse();

    // Bag of Words Representation
    void computeBoW();

    // Landmark observation functions
    void addLandmark(LandmarkPtr pMP, const size_t& idx);
    void eraseLandmarkMatch(const size_t& idx);
    void eraseLandmarkMatch(LandmarkPtr pMP);
    void replaceLandmarkMatch(const size_t& idx, LandmarkPtr pMP);
    std::set<LandmarkPtr> getLandmarks();
    std::vector<LandmarkPtr> getLandmarkMatches();
    int trackedLandmarks(const int& minObs);
    LandmarkPtr getLandmark(const size_t& idx);

    std::vector<size_t> getFeaturesInArea(const double& x, const double& y, const double& r) const;

    bool isInImage(const double& x, const double& y) const;

    // Enable/Disable bad flag changes
    void setNotErase();
    void setErase();

    // Set/check bad flag
    void setBadFlag();
    bool isBad();

    double getDepth(const size_t& i);

    static bool weightComp(int a, int b) { return a > b; }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) { return pKF1->_id < pKF2->_id; }

    // PointCloud operations
    void createPointCloud(const int& res = 3);
    void downsample(float leaf);
    void statisticalFilter(int k, double stddev);
    void passThroughFilter(const std::string& field, float ll, float ul, const bool negate = false);
    bool hasPointCloud();
    void createOctoCloud(PointCloudColor::Ptr worldCloud);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:
    cv::Mat _colorIm;
    cv::Mat _grayIm;
    cv::Mat _depthIm;

    PointCloudColor::Ptr _pointCloud;
    std::shared_ptr<octomap::Pointcloud> _octoCloud;

    static int _nextId;
    int _id;
    const int _frameId;

    const double _timestamp;

    // Grid (to speed up feature matching)
    const int _gridCols;
    const int _gridRows;
    const double _gridElementWidthInv;
    const double _gridElementHeightInv;

    // Variables used by the tracking
    int _trackReferenceForFrame;
    int _fuseTargetForKF;

    // Variables used by the local mapping
    int _BALocalForKF;
    int _BAFixedForKF;

    // Variables used by the keyframe database
    int _loopQuery;
    int _loopWords;
    double _loopScore;

    // Variables used by loop closing
    SE3 _TcwGBA;
    SE3 _TcwBefGBA;
    int _BAGlobalForKF;

    // Number of KeyPoints
    const size_t _N;

    // Features
    const std::vector<cv::KeyPoint> _keys;
    const cv::Mat _descriptors;
    const std::vector<FeaturePtr> _features;

    //BoW
    DBoW3::BowVector _bowVec;
    DBoW3::FeatureVector _featVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    SE3 _Tcp;

    // Scale
    const int _scaleLevels;
    const double _scaleFactor;
    const double _logScaleFactor;
    const std::vector<double> _scaleFactors;
    const std::vector<double> _levelSigma2;
    const std::vector<double> _invLevelSigma2;

    // Image bounds and calibration
    const int _minX;
    const int _minY;
    const int _maxX;
    const int _maxY;

    NodePtr _node;

    CameraPtr _camera;

    // The following variables need to be accessed trough a mutex to be thread safe.
protected:
    // SE3 Pose and camera center
    SE3 _Tcw;
    SE3 _Twc;

    // BoW
    KeyFrameDatabasePtr _KFDB;
    VocabularyPtr _vocabulary;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> _grid;

    std::vector<LandmarkPtr> _landmarks;

    // Bad flags
    bool _notErase;
    bool _toBeErased;
    bool _bad;

    MapPtr _map;

    std::mutex _mutexPose;
    std::mutex _mutexFeatures;
};

#endif // KEYFRAME_H
