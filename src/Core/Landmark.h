#ifndef LANDMARK_H
#define LANDMARK_H

#include "System/Common.h"
#include <map>
#include <mutex>
#include <opencv2/core/core.hpp>

class Landmark {
public:
    Landmark(const Vec3& Pos, KeyFrame* pRefKF, MapPtr pMap);
    Landmark(const Vec3& Pos, MapPtr pMap, Frame* pFrame, const int& idxF);

    void setWorldPos(const Vec3& Pos);
    Vec3 getWorldPos();

    Vec3 getNormal();
    KeyFrame* getReferenceKeyFrame();

    std::map<KeyFrame*, size_t> getObservations();
    int observations();

    void addObservation(KeyFrame* pKF, size_t idx);
    void eraseObservation(KeyFrame* pKF);

    int getIndexInKeyFrame(KeyFrame* pKF);
    bool isInKeyFrame(KeyFrame* pKF);

    void setBadFlag();
    bool isBad();

    void replace(LandmarkPtr pMP);
    LandmarkPtr getReplaced();

    void increaseVisible(int n = 1);
    void increaseFound(int n = 1);
    double getFoundRatio();
    inline int getFound() { return _found; }

    void computeDistinctiveDescriptors();

    cv::Mat getDescriptor();

    void updateNormalAndDepth();

    double getMinDistanceInvariance();
    double getMaxDistanceInvariance();
    int predictScale(const double& currentDist, KeyFrame* pKF);
    int predictScale(const double& currentDist, Frame* pF);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    int _id;
    static int _nextId;
    int _firstKFid;
    int _firstFrame;
    int _nobs;

    // Variables used by the tracking
    double _trackProjX;
    double _trackProjY;
    double _trackProjXR;
    bool _trackInView;
    int _trackScaleLevel;
    double _trackViewCos;
    int _trackReferenceForFrame;
    int _lastFrameSeen;

    // Variables used by local mapping
    int _BALocalForKF;
    int _fuseCandidateForKF;

    // Variables used by loop closing
    int _loopPointForKF;
    int _correctedByKF;
    int _correctedReference;
    Vec3 _posGBA;
    int _BAGlobalForKF;

    static std::mutex _globalMutex;

protected:
    // Position in absolute coordinates
    Vec3 _worldPos;

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame*, size_t> _observations;

    // Mean viewing direction
    Vec3 _normal;

    // Best descriptor to fast matching
    cv::Mat _descriptor;

    // Reference KeyFrame
    KeyFrame* _refKF;

    // Tracking counters
    int _visible;
    int _found;

    // Bad flag (we do not currently erase Landmark from memory)
    bool _bad;
    LandmarkPtr _replaced;

    // Scale invariance distances
    double _minDistance;
    double _maxDistance;

    MapPtr _map;

    std::mutex _mutexPos;
    std::mutex _mutexFeatures;
};

#endif
