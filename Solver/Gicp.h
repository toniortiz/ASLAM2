#ifndef GICP_H
#define GICP_H

#include "System/Common.h"
#include <opencv2/features2d.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class Frame;

class Gicp {
public:
    SMART_POINTER_TYPEDEFS(Gicp);

    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

public:
    Gicp(Frame* pF1, Frame* pF2, const std::vector<cv::DMatch>& vMatches);

    bool compute(cv::Mat& guess);

    Gicp& setCorrespondenceDistance(double dist);
    Gicp& setIterations(int iters);
    Gicp& setEuclideanEpsilon(double eps);
    Gicp& setTransformationEpsilon(double eps);

    double _score;
    cv::Mat _T;

protected:
    Frame* _frame1;
    Frame* _frame2;

    PointCloudT::Ptr _srcCloud;
    PointCloudT::Ptr _tgtCloud;

    // Parameters
    double _maxCorrespondenceDistance;
    int _iterations;
    double _euclideanEpsilon;
    double _transformationEpsilon;
};

#endif // GICP_H
