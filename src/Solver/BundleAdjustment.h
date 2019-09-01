#ifndef BUNDLEADJUSTMENT_H
#define BUNDLEADJUSTMENT_H

#include <vector>
#include "System/Common.h"
#include <g2o/types/sim3/types_seven_dof_expmap.h>

class BundleAdjustment {
public:
    static void Global(MapPtr pMap, int nIterations = 5, bool* pbStopFlag = nullptr,
        const int nLoopKF = 0, const bool bRobust = true);

    static void Local(KeyFrame* pKF, bool* pbStopFlag, MapPtr pMap);

    // if bFixScale is true, optimize SE3 rgbd
    static int OptimizeSE3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<LandmarkPtr>& vpMatches1,
                           g2o::Sim3& g2oS12, const double th2);

protected:
    static void GBA(const std::vector<KeyFrame*>& vpKFs, const std::vector<LandmarkPtr>& vpMPs,
        int nIterations = 5, bool* pbStopFlag = nullptr, const int nLoopKF = 0, const bool bRobust = true);
};

#endif // BUNDLEADJUSTMENT_H
