#include "PnP.h"
#include "Core/Frame.h"
#include "Core/Landmark.h"
#include "Core/PinholeCamera.h"
#include "System/Converter.h"
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>

using namespace std;

typedef opengv::absolute_pose::CentralAbsoluteAdapter Adapter;
typedef opengv::sac::Ransac<PnP::Problem> RANSAC;

PnP::PnP(Frame* pFrame)
    : _frame(pFrame)
{
    for (size_t i = 0; i < pFrame->_N; ++i) {
        Landmark* pMP = pFrame->_landmarks[i];
        if (!pMP)
            continue;

        Vec3 Xw = pMP->getWorldPos();
        Vec3 bv = pFrame->_camera->pixel2bearing(Vec2(pFrame->_keysUn[i].pt.x, pFrame->_keysUn[i].pt.y));

        _bearings.push_back(bv);
        _landmarks.push_back(Xw);
        _index.push_back(i);

        pFrame->setOutlier(i);
    }

    _focalLength = pFrame->_camera->fx();
    _twc = _frame->getPoseInverse().translation();
    _Rwc = _frame->getPoseInverse().rotationMatrix();

    setRansacParameters(1000, 1.0 - cos(atan(sqrt(2.0) * 0.5 / _focalLength)), 0.99);
    setAlgorithm(KNEIP);
    refine(true);
}

void PnP::setRansacParameters(int iterations, double threshold, double probability)
{
    _iterations = iterations;
    _threshold = threshold;
    _probability = probability;
}

PnP& PnP::setReprojectionTh(double pixels)
{
    _threshold = (1.0 - cos(atan(pixels / _focalLength)));
    return *this;
}

PnP& PnP::setAlgorithm(const PnP::eAlgorithm& algorithm)
{
    switch (algorithm) {
    case KNEIP:
        _algorithm = Problem::KNEIP;
        break;
    case GAO:
        _algorithm = Problem::GAO;
        break;
    case EPNP:
        _algorithm = Problem::EPNP;
        break;
    }

    return *this;
}

PnP& PnP::setIterations(int its)
{
    _iterations = its;
    return *this;
}

PnP& PnP::refine(bool b)
{
    _refinement = b;
    return *this;
}

bool PnP::compute()
{
    Adapter adapter(_bearings, _landmarks, _twc, _Rwc);
    shared_ptr<Problem> absposeproblem_ptr(new Problem(adapter, _algorithm));

    RANSAC ransac;
    ransac.sac_model_ = absposeproblem_ptr;
    ransac.probability_ = _probability;
    ransac.threshold_ = _threshold;
    ransac.max_iterations_ = _iterations;
    bool bOK = ransac.computeModel();

    opengv::transformation_t T = ransac.model_coefficients_;

    if (bOK && _refinement) {
        opengv::transformation_t nlT;
        absposeproblem_ptr->optimizeModelCoefficients(ransac.inliers_, T, nlT);
        T = nlT;
    }

    if (bOK) {
        Eigen::Matrix4d Rt;
        Rt << T(0, 0), T(0, 1), T(0, 2), T(0, 3),
            T(1, 0), T(1, 1), T(1, 2), T(1, 3),
            T(2, 0), T(2, 1), T(2, 2), T(2, 3),
            0, 0, 0, 1;

        // opengv give us Twc, we need Tcw
        _frame->setPose(SE3(Rt.inverse().matrix()));

        for (size_t i = 0; i < ransac.inliers_.size(); ++i)
            _frame->setInlier(_index[ransac.inliers_.at(i)]);

    } else {
        for (size_t i = 0; i < _frame->_N; ++i) {
            Landmark* pMP = _frame->_landmarks[i];
            if (!pMP)
                continue;
            _frame->setInlier(i);
        }
    }

    return bOK;
}
