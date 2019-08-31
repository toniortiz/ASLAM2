#ifndef POSEGRAPH_H
#define POSEGRAPH_H

#include "PlaceRecognition/LoopClosing.h"
#include "System/Common.h"

class PoseGraph {
public:
    static void optimize(MapPtr pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
        const LoopClosing::KeyFrameAndPose& NonCorrectedSE3,
        const LoopClosing::KeyFrameAndPose& CorrectedSE3,
        const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections);
};

#endif // POSEGRAPH_H
