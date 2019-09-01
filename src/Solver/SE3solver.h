#ifndef SE3SOLVER_H
#define SE3SOLVER_H

#include "System/Common.h"
#include <Eigen/Dense>
#include <opencv2/features2d.hpp>
#include <vector>

class Frame;
class KeyFrame;

class SE3solver {
public:
    SMART_POINTER_TYPEDEFS(SE3solver);

public:
    // For Tracking
    SE3solver(FramePtr pF1, FramePtr pF2, const std::vector<cv::DMatch>& vMatches12);

    // For loop detector
    SE3solver(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<cv::DMatch>& vMatches12);

    void setRansacParameters(int iters = 200, uint minInlierTh = 20, float maxMahalanobisDist = 3.0f, uint sampleSize = 4);

    SE3solver& setIterations(int iters);
    SE3solver& setMinInlierThreshold(uint th);
    SE3solver& setMaxMahalanobisDistance(float md);
    SE3solver& setSampleSize(uint ss);

    bool compute();

    float _rmse;
    std::vector<cv::DMatch> _inliers;
    SE3 _T;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    std::vector<cv::DMatch> sampleMatches();

    Eigen::Matrix4f computeHypothesis(const std::vector<cv::DMatch>& vMatches, bool& valid);

    double computeError(const Eigen::Matrix4f& transformation4f, std::vector<cv::DMatch>& vInlierMatches);

    double errorFunction2(const Eigen::Vector4f& x1, const Eigen::Vector4f& x2, const Eigen::Matrix4d& transformation);

    double depthCovariance(double depth);

    double depthStdDev(double depth);

    const std::vector<FeaturePtr> _vC1;
    const std::vector<FeaturePtr> _vC2;
    std::vector<cv::DMatch> _matchesToUse;

    int _iterations;
    uint _minInlierTh;
    float _maxMahalanobisDistance;
    uint _sampleSize;

    // For Tracking
    FramePtr _frame1;
    FramePtr _frame2;
};

#endif // SE3SOLVER_H
