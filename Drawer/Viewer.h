#ifndef VIEWER_H
#define VIEWER_H

#include "System/Common.h"
#include <mutex>
#include <thread>
#include <vector>

class Viewer {
public:
    SMART_POINTER_TYPEDEFS(Viewer);

    Viewer(System* pSystem, MapDrawerPtr pMapDrawer, TrackingPtr pTracking, MapPtr pMap,
        DenseMapPtr denseMap, const bool& start = true);

    void start();

    // Main thread function. Draw points, keyframes, the current camera pose and the last processed
    // frame. Drawing is refreshed according to the camera fps. We use Pangolin.
    void run();

    void requestFinish();

    void requestStop();

    bool isFinished();

    bool isStopped();

    void release();

    void join();

    void setMeanTime(const double& time);
    void setRansacError(double rmse);
    void setIcpScore(double score);

private:
    bool stop();

    System* _system;
    MapDrawerPtr _mapDrawer;
    DenseMapPtr _denseMap;
    TrackingPtr _tracker;
    MapPtr _map;

    float _imageWidth, _imageHeight;

    float _viewpointX, _viewpointY, _viewpointZ, _viewpointF;

    bool checkFinish();
    void setFinish();
    bool _finishRequested;
    bool _finished;
    std::mutex _mutexFinish;

    bool _stopped;
    bool _stopRequested;
    std::mutex _mutexStop;

    double _meanTrackTime;
    double _rmseSac;
    double _scoreIcp;
    std::mutex _mutexStatistics;

    std::thread _thread;
};

#endif // VIEWER_H
