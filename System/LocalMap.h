#ifndef LOCALMAP_H
#define LOCALMAP_H

#include "System/Common.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <opencv2/core.hpp>

class LocalMap {
public:
    SMART_POINTER_TYPEDEFS(LocalMap);

    LocalMap(CameraPtr cam);

    ~LocalMap();

    bool track(FramePtr frame);

    int optimize();

    void release();

    std::vector<Landmark*> _localPoints;
    std::vector<KeyFrame*> _localKFs;

    KeyFrame* _refKF;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    void searchLocalKFs();
    void searchLocalPoints();
    void projectLocalPoints();

    g2o::VertexSE3Expmap* addVertex(const SE3& Tcw, int id, bool fixed);
    g2o::EdgeSE3ProjectXYZOnlyPose* addMonoEdge(const FeaturePtr kp, Landmark* pMP);
    g2o::EdgeStereoSE3ProjectXYZOnlyPose* addStereoEdge(const FeaturePtr kpUn, Landmark* pMP);

    g2o::SparseOptimizer* _optimizer;

    // Reference data
    FramePtr _curFrame;

    // Calibration
    CameraPtr _camera;
    double _fx, _fy, _cx, _cy, _bf;

    // Data association
    const double DELTA_MONO;
    const double DELTA_STEREO;
};

#endif // LOCALMAP_H
