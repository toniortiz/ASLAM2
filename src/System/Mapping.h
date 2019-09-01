#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "Common.h"
#include <list>
#include <mutex>
#include <opencv2/core.hpp>
#include <thread>

class Mapping {
public:
    SMART_POINTER_TYPEDEFS(Mapping);

    Mapping(MapPtr pMap, const bool& start = true);

    void setLoopCloser(LoopClosingPtr pLoopCloser);

    void setTracker(TrackingPtr pTracker);

    void start();

    // Main function
    void run();

    void insertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void requestStop();
    void requestReset();
    bool stop();
    void release();
    bool isStopped();
    bool stopRequested();
    bool acceptKeyFrames();
    void setAcceptKeyFrames(bool flag);
    bool setNotStop(bool flag);

    void interruptBA();

    void requestFinish();
    bool isFinished();

    int KeyframesInQueue()
    {
        std::unique_lock<std::mutex> lock(_mutexQueue);
        return _KFsQueue.size();
    }

    void join();

protected:
    bool checkNewKeyFrames();
    void processNewKeyFrame();

    void landmarkCulling();
    void searchInNeighbors();

    void KeyFrameCulling();

    void resetIfRequested();
    bool _resetRequested;
    std::mutex _mutexReset;

    bool checkFinish();
    void setFinish();
    bool _finishRequested;
    bool _finished;
    std::mutex _mutexFinish;

    MapPtr _map;

    LoopClosingPtr _loopCloser;
    TrackingPtr _tracker;

    std::list<KeyFrame*> _KFsQueue;

    KeyFrame* _curKF;

    std::list<LandmarkPtr> _recentAddedLandmarks;

    std::mutex _mutexQueue;

    bool _abortBA;

    bool _stopped;
    bool _stopRequested;
    bool _notStop;
    std::mutex _mutexStop;

    bool _acceptKFs;
    std::mutex _mutexAccept;

    std::thread _thread;
};

#endif // LOCALMAPPING_H
