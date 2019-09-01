#ifndef SYSTEM_H
#define SYSTEM_H

#include "Common.h"
#include <DBoW3/Vocabulary.h>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <string>
#include <thread>

class System {
public:
    System(ExtractorPtr pExtractor, VocabularyPtr pVoc, DatasetPtr pDataset, const bool bUseViewer = true, const bool bUseDenseMap=false);

    ~System();

    SE3 trackRGBD(const cv::Mat& im, const cv::Mat& depthmap, const double& timestamp);

    // Returns true if there have been a big map change (loop closure, global BA)
    bool mapChanged();

    // Reset the system (clear map)
    void reset();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void shutdown();

    void saveTrajectoryTUM(const std::string& filename);

    void saveKeyFrameTrajectoryTUM(const std::string& filename);

    void saveObservationHistogram(const std::string& filename);

    void saveCovisibilityGraph(const std::string& filename);

    void setMeanTime(const double& time);

    void informBigChange();

private:
    // Place recognition
    VocabularyPtr _vocabulary;
    KeyFrameDatabasePtr _KeyFrameDatabase;

    MapPtr _map;
    TrackingPtr _tracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    MappingPtr _localMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosingPtr _loopCloser;

    ViewerPtr _viewer;
    MapDrawerPtr _mapDrawer;
    DenseMapPtr _denseMap;

    // Reset flag
    std::mutex _mutexReset;
    bool _reset;
};

#endif // SYSTEM_H
