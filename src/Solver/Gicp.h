#ifndef GICP_H
#define GICP_H

#include "System/Common.h"
#include <opencv2/features2d.hpp>

class Gicp {
public:
    SMART_POINTER_TYPEDEFS(Gicp);

    Gicp(FramePtr pF1, FramePtr pF2, const std::vector<cv::DMatch>& vMatches);

    bool compute(const SE3& guess);

    Gicp& setCorrespondenceDistance(double dist);
    Gicp& setIterations(int iters);
    Gicp& setEuclideanEpsilon(double eps);
    Gicp& setTransformationEpsilon(double eps);

    double _score;
    SE3 _T;

private:
    void computeNormals(PointCloud::Ptr cloud,PointCloudNormal::Ptr normals, double radius);

    FramePtr _frame1;
    FramePtr _frame2;

    PointCloud::Ptr _srcCloud;
    PointCloudNormal::Ptr _srcNormals;
    PointCloud::Ptr _tgtCloud;
    PointCloudNormal::Ptr _tgtNormals;

    // Parameters
    double _maxCorrespondenceDistance;
    int _iterations;
    double _euclideanEpsilon;
    double _transformationEpsilon;
};

#endif // GICP_H
