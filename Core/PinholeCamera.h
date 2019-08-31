#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

#include "System/Common.h"
#include <opencv2/core.hpp>

class PinholeCamera {
public:
    SMART_POINTER_TYPEDEFS(PinholeCamera);

public:
    PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
        const double bf, const double thDepth, const double depthFactor, const double fps,
        double k1 = 0.0, double k2 = 0.0, double k3 = 0.0, double p1 = 0.0, double p2 = 0.0);

    PinholeCamera();

    PinholeCamera* clone() const;

    void undistortImage(const cv::Mat& raw, cv::Mat& rectified) const;
    Vec2 undistortPoint(const Vec2& raw) const;
    void undistortBounds(double& minX, double& maxX, double& minY, double& maxY) const;

    Vec3 pixel2bearing(const double& x, const double& y) const;
    Vec3 pixel2bearing(const Vec2& pi) const;

    // Porject a 3D point in camera coordinates to 2D pixel coordinates
    Vec2 project(const Vec3& pc) const;

    // Backproject a 2D pixel to 3D camera coordinates
    Vec3 backproject(const double& x, const double& y, const double& depth) const;
    Vec3 backproject(const Vec2& pi, const double& depth) const;

    void print(std::ostream& out, const std::string& text = "") const;

    bool hasDistortion() const;

    // Return calibration matrix
    Mat33 K() const;

    // Return distortion vector
    VecX D() const;

    double fx() const;
    double fy() const;
    double cx() const;
    double cy() const;

    double depthFactor() const;

    double maxDepthTh() const;
    double baseLineFx() const;
    double baseLine() const;

    friend std::ostream& operator<<(std::ostream& out, const PinholeCamera& cam);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
    int _width;
    int _height;

    // Calibration matrix
    double _fx, _fy, _cx, _cy;
    double _invfx, _invfy;
    Mat33 _K;

    // Stereo baseline in meters.
    double _b;

    double _fps;

    // For some datasets (e.g. TUM) the depthmap values are scaled.
    double _depthFactor;

    // Stereo baseline multiplied by fx.
    double _bf;

    // Threshold close/far points.
    double _thDepth;

    // Distortion vector
    double _k1, _k2, _k3, _p1, _p2;
    VecX _D;

    cv::Mat _cvK;
    cv::Mat _cvD;
    cv::Mat _umap1;
    cv::Mat _umap2;
};

#endif // PINHOLECAMERA_H
