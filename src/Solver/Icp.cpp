#include "Icp.h"
#include "Core/Frame.h"
#include "Core/KeyFrame.h"
#include "System/Converter.h"
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include "Core/Feature.h"
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>

using namespace std;

typedef opengv::point_cloud::PointCloudAdapter Adapter;
typedef opengv::sac_problems::point_cloud::PointCloudSacProblem ProblemICP;
typedef opengv::sac::Ransac<ProblemICP> RANSAC;

Icp::Icp(Frame* pF1, Frame* pF2, const vector<cv::DMatch>& vMatches)
    : _frame1(pF1)
    , _frame2(pF2)
{
    _points1.reserve(vMatches.size());
    _points2.reserve(vMatches.size());
    _matches.reserve(vMatches.size());

    for (const auto& m : vMatches) {
        const Vec3& p1 = pF1->_features[m.queryIdx]->_Xc;
        _points1.push_back(p1);

        const Vec3& p2 = pF2->_features[m.trainIdx]->_Xc;
        _points2.push_back(p2);

        _matches.push_back(m);
        pF2->_features[m.trainIdx]->setOutlier();
    }

    setRansacParameters(1000, 0.07, 0.99);
}

Icp::Icp(KeyFrame* pKF1, KeyFrame* pKF2, const vector<cv::DMatch>& vMatches)
    : _frame1(nullptr)
    , _frame2(nullptr)
{
    _points1.reserve(vMatches.size());
    _points2.reserve(vMatches.size());
    _matches.reserve(vMatches.size());

    for (const auto& m : vMatches) {
        const Vec3& p1 = pKF1->_features[m.queryIdx]->_Xc;
        _points1.push_back(p1);

        const Vec3& p2 = pKF2->_features[m.trainIdx]->_Xc;
        _points2.push_back(p2);

        _matches.push_back(m);
    }

    setRansacParameters(1000, 0.07, 0.99);
    refine(true);
}

void Icp::setRansacParameters(int iterations, double threshold, double probability)
{
    _maxIterations = iterations;
    _threshold = threshold;
    _probability = probability;
}

Icp& Icp::setThreshold(double th)
{
    _threshold = th;
    return *this;
}

Icp& Icp::setIterations(int iters)
{
    _maxIterations = iters;
    return *this;
}

Icp& Icp::setProbability(double prob)
{
    _probability = prob;
    return *this;
}

Icp& Icp::refine(bool r)
{
    _refinement = r;
    return *this;
}

bool Icp::compute()
{
    Adapter adapter(_points2, _points1);
    shared_ptr<ProblemICP> icpproblem_ptr(new ProblemICP(adapter));

    RANSAC ransac;
    ransac.sac_model_ = icpproblem_ptr;
    ransac.probability_ = _probability;
    ransac.max_iterations_ = _maxIterations;
    ransac.threshold_ = _threshold;

    bool bOK = ransac.computeModel();

    _inliers.clear();
    _T = SE3(Mat44::Identity());

    if (bOK) {
        opengv::transformation_t T = ransac.model_coefficients_;
        _inliers.reserve(ransac.inliers_.size());

        if (_refinement) {
            opengv::transformation_t nlT;
            icpproblem_ptr->optimizeModelCoefficients(ransac.inliers_, T, nlT);
            T = nlT;
        }

        _T = SE3(T.leftCols(3), T.rightCols(1));

        if (_frame2) {
            for (size_t i = 0; i < ransac.inliers_.size(); ++i) {
                _frame2->_features[_matches[ransac.inliers_.at(i)].trainIdx]->setInlier();
                _inliers.push_back(_matches[ransac.inliers_.at(i)]);
            }

            _frame2->setPose(_T * _frame1->getPose());
        } else {
            for (size_t i = 0; i < ransac.inliers_.size(); ++i)
                _inliers.push_back(_matches[ransac.inliers_.at(i)]);
        }
    } else {
        if (_frame2) {
            for (const auto& m : _matches)
                _frame2->_features[m.trainIdx]->setInlier();
        }
    }

    return bOK;
}
