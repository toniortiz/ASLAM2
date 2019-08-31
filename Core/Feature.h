#ifndef FEATURE_H
#define FEATURE_H

#include "System/Common.h"

class Feature {
public:
    Feature(Frame* frame, const Vec2& px, const int level, const double angle, const cv::Mat& desc, size_t i);

    bool isValid() const;

    void setInlier();
    void setOutlier();
    bool isInlier() const;
    bool isOutlier() const;

    double getDepth() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    size_t _index;

    // Pixel coordinates
    Vec2 _Xi;

    // Undistorted pixel coordinate
    Vec2 _uXi;

    // 3D point in camera coordinates
    Vec3 _Xc;

    // Descriptor associated
    cv::Mat _d;

    // Unit-bearing vector of the feature
    Vec3 _bv;

    // Image pyramid level where feature was extracted
    int _level;

    double _angle;

    // Stereo coordinate
    double _right;

    // Dominant gradient direction
    Vec2 _grad;

    bool _outlier;

    Landmark* _point;
};

#endif // FEATURE_H
