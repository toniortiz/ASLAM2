#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "System/Common.h"
#include <DBoW3/Vocabulary.h>
#include <Eigen/Dense>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <mutex>
#include <set>
#include <thread>

class LoopClosing {
public:
    SMART_POINTER_TYPEDEFS(LoopClosing);

    typedef std::pair<std::set<KeyFrame*>, int> ConsistentGroup;
    typedef std::map<KeyFrame*, g2o::SE3Quat, std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::SE3Quat>>>
        KeyFrameAndPose;

    LoopClosing(MapPtr pMap, KeyFrameDatabasePtr pDB, VocabularyPtr pVoc, const bool& start = true);

    void setTracker(TrackingPtr pTracker);

    void setLocalMapper(MappingPtr pLocalMapper);

    void start();

    // Main function
    void run();

    void insertKeyFrame(KeyFrame* pKF);

    void requestReset();

    // This function will run in a separate thread
    void runGlobalBundleAdjustment(int nLoopKF);

    bool isRunningGBA()
    {
        std::unique_lock<std::mutex> lock(_mutexGBA);
        return _runningGBA;
    }
    bool isFinishedGBA()
    {
        std::unique_lock<std::mutex> lock(_mutexGBA);
        return _finishedGBA;
    }

    void requestFinish();

    bool isFinished();

    void join();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    bool checkNewKeyFrames();

    bool detectLoop();

    bool computeSE3();

    void searchAndFuse(const KeyFrameAndPose& CorrectedPosesMap);

    void correctLoop();

    void resetIfRequested();
    bool _resetRequested;
    std::mutex _mutexReset;

    bool checkFinish();
    void setFinish();
    bool _finishRequested;
    bool _finished;
    std::mutex _mutexFinish;

    MapPtr _map;
    TrackingPtr _tracker;

    KeyFrameDatabasePtr _KFDB;
    VocabularyPtr _vocabulary;

    MappingPtr _localMapper;

    std::list<KeyFrame*> _KFsQueue;

    std::mutex _mutexQueue;

    // Loop detector parameters
    float _covisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame* _curKF;
    KeyFrame* _matchedKF;
    std::vector<ConsistentGroup> _consistentGroups;
    std::vector<KeyFrame*> _enoughConsistentCandidates;
    std::vector<KeyFrame*> _curConnectedKFs;
    std::vector<LandmarkPtr> _curMatchedPoints;
    std::vector<LandmarkPtr> _loopLandmarks;
    SE3 _Tcw;
    g2o::SE3Quat _g2oTcw;

    int _prevLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool _runningGBA;
    bool _finishedGBA;
    bool _stopGBA;
    std::mutex _mutexGBA;
    std::thread* _threadGBA;

    int _fullBAIdx;

    std::thread _thread;
};

#endif // LOOPCLOSING_H
