#ifndef ICP_H
#define ICP_H

#include "System/Common.h"
#include <opencv2/features2d.hpp>
#include <vector>

class Frame;
class KeyFrame;

class Icp {
public:
    SMART_POINTER_TYPEDEFS(Icp);

    typedef opengv::points_t Points3Dvector;
    typedef opengv::point_t Point3D;

public:
    // For Tracking
    Icp(Frame* pF1, Frame* pF2, const std::vector<cv::DMatch>& vMatches);

    // For Loop Closing
    Icp(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<cv::DMatch>& vMatches);

    void setRansacParameters(int iterations, double threshold, double probability);

    Icp& setThreshold(double th);
    Icp& setIterations(int iters);
    Icp& setProbability(double prob);
    Icp& refine(bool r);

    bool compute();

    std::vector<cv::DMatch> _inliers;
    SE3 _T;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    Frame* _frame1;
    Frame* _frame2;

    std::vector<Vec3> _points1;
    std::vector<Vec3> _points2;
    std::vector<cv::DMatch> _matches;

    // Ransac parameters
    double _threshold;
    double _probability;
    int _maxIterations;
    bool _refinement;
};
#endif // ICP_H
