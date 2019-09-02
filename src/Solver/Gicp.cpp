#include "Gicp.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include "System/Converter.h"
#include <pcl/features/normal_3d.h>
#include <pcl/registration/gicp.h>
#include <thread>

using namespace std;

Gicp::Gicp(FramePtr pF1, FramePtr pF2, const vector<cv::DMatch>& vMatches)
    : _frame1(pF1)
    , _frame2(pF2)
{
    _srcCloud = boost::make_shared<PointCloud>();
    _tgtCloud = boost::make_shared<PointCloud>();

    _srcCloud->points.reserve(vMatches.size());
    _tgtCloud->points.reserve(vMatches.size());

    for (const auto& m : vMatches) {
        const Vec3& p1 = pF1->_features[m.queryIdx]->_Xc;
        _srcCloud->points.push_back(Point(p1.x(), p1.y(), p1.z()));

        const Vec3& p2 = pF2->_features[m.trainIdx]->_Xc;
        _tgtCloud->points.push_back(Point(p2.x(), p2.y(), p2.z()));
    }

    _srcNormals.reset(new PointCloudNormal);
    _tgtNormals.reset(new PointCloudNormal);
    _srcNormals->points.reserve(_srcCloud->points.size());
    _tgtNormals->points.reserve(_tgtCloud->points.size());

    double radius = 0.25;
    thread t1(&Gicp::computeNormals, this, _srcCloud, _srcNormals, radius);
    thread t2(&Gicp::computeNormals, this, _tgtCloud, _tgtNormals, radius);
    t1.join();
    t2.join();

    // Default parameters
    _iterations = 15;
    _maxCorrespondenceDistance = 0.07;
    _euclideanEpsilon = 1;
    _transformationEpsilon = 1e-9;
}

bool Gicp::compute(const SE3& guess)
{
    _score = numeric_limits<double>::max();
    _T = SE3(Mat44::Identity());

    if (_srcNormals->points.size() < 20)
        return false;

    pcl::GeneralizedIterativeClosestPoint<PointNormal, PointNormal> gicp;
    gicp.setInputSource(_srcNormals);
    gicp.setInputTarget(_tgtNormals);
    gicp.setMaximumIterations(_iterations);
    gicp.setMaxCorrespondenceDistance(_maxCorrespondenceDistance);
    gicp.setEuclideanFitnessEpsilon(_euclideanEpsilon);
    gicp.setTransformationEpsilon(_transformationEpsilon);

    PointCloudNormal aligned;
    gicp.align(aligned, guess.matrix().cast<float>());

    if (gicp.hasConverged()) {
        _score = gicp.getFitnessScore();
        _T = SE3(gicp.getFinalTransformation().cast<double>());
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

void Gicp::computeNormals(PointCloud::Ptr cloud, PointCloudNormal::Ptr normals, double radius)
{
    pcl::NormalEstimation<Point, PointNormal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);
    ne.compute(*normals);
}
