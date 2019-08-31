#include "PinholeCamera.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

PinholeCamera::PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
    const double bf, const double thDepth, const double depthFactor, const double fps,
    double k1, double k2, double k3, double p1, double p2)
    : _width(width)
    , _height(height)
    , _fx(fx)
    , _fy(fy)
    , _cx(cx)
    , _cy(cy)
    , _fps(fps)
    , _bf(bf)
    , _k1(k1)
    , _k2(k2)
    , _k3(k3)
    , _p1(p1)
    , _p2(p2)
{
    _K = Mat33::Identity();
    _K(0, 0) = _fx;
    _K(1, 1) = _fy;
    _K(0, 2) = _cx;
    _K(1, 2) = _cy;

    _invfx = 1.0 / _fx;
    _invfy = 1.0 / _fy;

    if (k1 != 0.0) {
        _D.resize(4);
        _D[0] = _k1;
        _D[1] = _k2;
        _D[2] = _p1;
        _D[3] = _p2;
        if (_k3 != 0.0) {
            _D.resize(5);
            _D[4] = _k3;
        }
    } else {
        _D.resize(0);
    }

    cv::eigen2cv(_K, _cvK);
    cv::eigen2cv(_D, _cvD);

    if (hasDistortion()) {
        _umap1 = cv::Mat(int(_height), int(_width), CV_16SC2);
        _umap2 = cv::Mat(int(_height), int(_width), CV_16SC2);
        cv::initUndistortRectifyMap(_cvK, _cvD, cv::Mat_<double>::eye(3, 3), _cvK, cv::Size(int(_width), int(_height)), CV_16SC2, _umap1, _umap2);
    }

    _thDepth = _bf * thDepth / _fx;
    _b = _bf / _fx;
    _depthFactor = 1.0 / depthFactor;
}

PinholeCamera::PinholeCamera() {}

PinholeCamera* PinholeCamera::clone() const
{
    PinholeCamera* pc = new PinholeCamera();
    pc->_width = _width;
    pc->_height = _height;
    pc->_fx = _fx;
    pc->_fy = _fy;
    pc->_cx = _cx;
    pc->_cy = _cy;
    pc->_K = _K;
    pc->_invfx = _invfx;
    pc->_invfy = _invfy;
    pc->_k1 = _k1;
    pc->_k2 = _k2;
    pc->_k3 = _k3;
    pc->_p1 = _p1;
    pc->_p2 = _p2;
    pc->_D = _D;
    pc->_cvK = _cvK.clone();
    pc->_cvD = _cvD.clone();
    pc->_umap1 = _umap1.clone();
    pc->_umap2 = _umap2.clone();
    pc->_fps = _fps;
    pc->_bf = _bf;
    pc->_thDepth = _thDepth;
    pc->_b = _b;
    pc->_depthFactor = _depthFactor;

    return pc;
}

void PinholeCamera::undistortImage(const cv::Mat& raw, cv::Mat& rectified) const
{
    if (hasDistortion()) {
        cv::remap(raw, rectified, _umap1, _umap2, CV_INTER_LINEAR);
    } else {
        rectified = raw.clone();
    }
}

Vec2 PinholeCamera::undistortPoint(const Vec2& raw) const
{
    if (hasDistortion()) {
        cv::Mat mat(1, 2, CV_64F);
        mat.at<double>(0, 0) = raw.x();
        mat.at<double>(0, 1) = raw.y();

        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, _cvK, _cvD, cv::Mat(), _cvK);
        mat = mat.reshape(1);

        Vec2 rect;
        rect[0] = mat.at<double>(0, 0);
        rect[1] = mat.at<double>(0, 1);
        return rect;
    } else {
        return raw;
    }
}

void PinholeCamera::undistortBounds(double& minX, double& maxX, double& minY, double& maxY) const
{
    if (hasDistortion()) {
        cv::Mat mat(4, 2, CV_64F);
        mat.at<double>(0, 0) = 0.0;
        mat.at<double>(0, 1) = 0.0;
        mat.at<double>(1, 0) = _width;
        mat.at<double>(1, 1) = 0.0;
        mat.at<double>(2, 0) = 0.0;
        mat.at<double>(2, 1) = _height;
        mat.at<double>(3, 0) = _width;
        mat.at<double>(3, 1) = _height;

        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, _cvK, _cvD, cv::Mat(), _cvK);
        mat = mat.reshape(1);

        minX = min(mat.at<double>(0, 0), mat.at<double>(2, 0));
        maxX = max(mat.at<double>(1, 0), mat.at<double>(3, 0));
        minY = min(mat.at<double>(0, 1), mat.at<double>(1, 1));
        maxY = max(mat.at<double>(2, 1), mat.at<double>(3, 1));
    } else {
        minX = 0.0;
        maxX = _width;
        minY = 0.0;
        maxY = _height;
    }
}

Vec3 PinholeCamera::pixel2bearing(const double& x, const double& y) const
{
    Vec3 bearing;
    bearing[0] = (x - _cx) * _invfx;
    bearing[1] = (y - _cy) * _invfy;
    bearing[2] = 1.0;
    return bearing.normalized();
}

Vec3 PinholeCamera::pixel2bearing(const Vec2& pi) const
{
    return pixel2bearing(pi[0], pi[1]);
}

Vec2 PinholeCamera::project(const Vec3& pc) const
{
    const double& x = pc[0];
    const double& y = pc[1];
    const double& z = pc[2];

    const double invz = 1.0 / z;
    double u = _fx * x * invz + _cx;
    double v = _fy * y * invz + _cy;
    return Vec2(u, v);
}

Vec3 PinholeCamera::backproject(const double& x, const double& y, const double& depth) const
{
    Vec3 pc;
    pc[0] = (x - _cx) * depth * _invfx;
    pc[1] = (y - _cy) * depth * _invfy;
    pc[2] = depth;
    return pc;
}

Vec3 PinholeCamera::backproject(const Vec2& pi, const double& depth) const
{
    return backproject(pi[0], pi[1], depth);
}

void PinholeCamera::print(ostream& out, const string& text) const
{
    if (!text.empty())
        out << text << endl;
    out << " - K:\n    " << _K << endl;
    if (hasDistortion())
        out << " -D: " << _D.transpose() << endl;
}

bool PinholeCamera::hasDistortion() const
{
    return _D.size() > 0;
}

Mat33 PinholeCamera::K() const
{
    return _K;
}

VecX PinholeCamera::D() const
{
    return _D;
}

double PinholeCamera::fx() const
{
    return _fx;
}

double PinholeCamera::fy() const
{
    return _fy;
}

double PinholeCamera::cx() const
{
    return _cx;
}

double PinholeCamera::cy() const
{
    return _cy;
}
double PinholeCamera::maxDepthTh() const
{
    return _thDepth;
}

double PinholeCamera::baseLineFx() const
{
    return _bf;
}

double PinholeCamera::baseLine() const
{
    return _b;
}

double PinholeCamera::depthFactor() const
{
    return _depthFactor;
}

ostream& operator<<(ostream& out, const PinholeCamera& cam)
{
    cam.print(out);
    return out;
}
