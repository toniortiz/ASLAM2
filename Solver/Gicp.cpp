#include "Gicp.h"
#include "Core/Frame.h"
#include "System/Converter.h"
#include <pcl/registration/gicp.h>

using namespace std;

Gicp::Gicp(Frame* pF1, Frame* pF2, const vector<cv::DMatch>& vMatches)
    : _frame1(pF1)
    , _frame2(pF2)
{
    _srcCloud = boost::make_shared<PointCloudT>();
    _tgtCloud = boost::make_shared<PointCloudT>();

    _srcCloud->points.reserve(vMatches.size());
    _tgtCloud->points.reserve(vMatches.size());

    for (const auto& m : vMatches) {
        const cv::Point3f& p1 = pF1->_camPoints[m.queryIdx];
        _srcCloud->points.push_back(PointT(p1.x, p1.y, p1.z));

        const cv::Point3f& p2 = pF2->_camPoints[m.trainIdx];
        _tgtCloud->points.push_back(PointT(p2.x, p2.y, p2.z));
    }

    // Default parameters
    _iterations = 15;
    _maxCorrespondenceDistance = 0.07;
    _euclideanEpsilon = 1;
    _transformationEpsilon = 1e-9;
}

bool Gicp::compute(cv::Mat& guess)
{
    _score = numeric_limits<double>::max();
    _T = cv::Mat();

    if (_srcCloud->points.size() < 20)
        return false;

    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    gicp.setInputSource(_srcCloud);
    gicp.setInputTarget(_tgtCloud);
    gicp.setMaximumIterations(_iterations);
    gicp.setMaxCorrespondenceDistance(_maxCorrespondenceDistance);
    gicp.setEuclideanFitnessEpsilon(_euclideanEpsilon);
    gicp.setTransformationEpsilon(_transformationEpsilon);

    PointCloudT aligned;
    gicp.align(aligned, Converter::toMatrix4f(guess));

    if (gicp.hasConverged()) {
        _score = gicp.getFitnessScore();
        _T = Converter::toMat<float, 4, 4>(gicp.getFinalTransformation());
        _frame2->setPose(_T * _frame1->getPose());
        return true;
    } else
        return false;
}

Gicp& Gicp::setCorrespondenceDistance(double dist)
{
    _maxCorrespondenceDistance = dist;
    return *this;
}

Gicp& Gicp::setIterations(int iters)
{
    _iterations = iters;
    return *this;
}

Gicp& Gicp::setEuclideanEpsilon(double eps)
{
    _euclideanEpsilon = eps;
    return *this;
}

Gicp& Gicp::setTransformationEpsilon(double eps)
{
    _transformationEpsilon = eps;
    return *this;
}
