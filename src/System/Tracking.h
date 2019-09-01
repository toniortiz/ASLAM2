#ifndef TRACKING_H
#define TRACKING_H

#include "System/Common.h"
#include <mutex>
#include <opencv2/core/core.hpp>

struct Statistics {
    std::mutex mMutex;

    int acumInliers = 0;
    int meanInliers = 0;
    int acumObservations = 0;
    int meanObservations = 0;
    int acumMatches = 0;
    int meanMatches = 0;
    double acumInlierRatio = 0;
    double meanInlierRatio = 0;

    // Local Map
    int nLocalMPs = 0;
    int nLocalKFs = 0;
};

class Tracking {
public:
    SMART_POINTER_TYPEDEFS(Tracking);

    Tracking(System* pSys, ExtractorPtr pExtractor, VocabularyPtr pVoc, MapDrawerPtr pMapDrawer, MapPtr pMap,
        KeyFrameDatabasePtr pKFDB, DatasetPtr pDataset);

    SE3 grabImageRGBD(const cv::Mat& imBGR, const cv::Mat& imD, const double& timestamp);

    void setLocalMapper(MappingPtr pLocalMapper);
    void setLoopClosing(LoopClosingPtr pLoopClosing);
    void setViewer(ViewerPtr pViewer);

    int meanObservations();
    int meanInliers();
    double meanInlierRatio();
    int meanMatches();
    int localMPs();
    int localKFs();

    cv::Mat getImageMatches();

public:
    // Tracking states
    enum eTrackingState {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    eTrackingState _state;

    // Current Frame
    FramePtr _curFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    std::list<SE3> _relativeFramePoses;
    std::list<KeyFrame*> _references;
    std::list<double> _frameTimes;
    std::list<bool> _losts;

    void reset();

private:
    // Main tracking function.
    void track();

    void updateMotionModel();
    void cleanOdometryMatches();
    void updateRelativePoses();

    void initialization();

    void checkReplacedInLastFrame();
    bool trackReferenceKeyFrame();
    void updatePrevFrame();
    bool trackWithMotionModel();

    void createNewLandmarks(FramePtr frame, KeyFrame* pKF = nullptr);

    bool needNewKeyFrame();
    void createNewKeyFrame();

    MappingPtr _localMapper;
    LoopClosingPtr _loopClosing;

    ExtractorPtr _extractor;

    // BoW
    VocabularyPtr _vocabulary;
    KeyFrameDatabasePtr _KFDB;

    System* _system;

    // Drawers
    ViewerPtr _viewer;
    MapDrawerPtr _mapDrawer;

    LocalMapPtr _localMap;
    MapPtr _map;

    // Last Frame, KeyFrame info
    KeyFrame* _prevKF;
    FramePtr _prevFrame;
    int _prevKFid;

    // Motion Model
    SE3 _speed;

    // Statistics
    Statistics _stats;

    CameraPtr _camera;

    cv::Mat _imgMatches;
    std::mutex _mutexImages;
};

#endif // TRACKING_H
