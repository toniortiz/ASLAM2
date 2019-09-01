#include "Feature.h"
#include "Frame.h"
#include "Landmark.h"
#include "PinholeCamera.h"

using namespace std;

Feature::Feature(Frame* frame, const Vec2& px, const int level, const double angle, const cv::Mat& desc, size_t i)
    : _index(i)
    , _Xi(px)
    , _d(desc)
    , _level(level)
    , _angle(angle)
    , _right(-1)
    , _grad(1.0, 0.0)
    , _outlier(false)
    , _point(nullptr)
{
    _uXi = frame->_camera->undistortPoint(_Xi);
    _bv = frame->_camera->pixel2bearing(_uXi);
    double d = frame->getDepth(_Xi);
    _Xc = frame->_camera->backproject(_uXi, d);
    _right = _uXi.x() - frame->_camera->baseLineFx() / d;
}

bool Feature::isValid() const
{
    return _Xc.z() > 0.0;
}

void Feature::setInlier()
{
    _outlier = false;
}

void Feature::setOutlier()
{
    _outlier = true;
}

bool Feature::isInlier() const
{
    return _outlier == false;
}

bool Feature::isOutlier() const
{
    return _outlier == true;
}

double Feature::getDepth() const
{
    return _Xc.z();
}
