#include "PnPRansac.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include "Core/Landmark.h"
#include "Core/PinholeCamera.h"
#include "System/Converter.h"

using namespace std;

PnPRansac::PnPRansac(FramePtr pFrame)
    : _frame(pFrame)
{
    for (size_t i = 0; i < pFrame->_N; ++i) {
        LandmarkPtr pMP = pFrame->_features[i]->_point;
        if (!pMP)
            continue;

        const Vec2& kpU = pFrame->_features[i]->_uXi;
        Vec3 Xw = pMP->getWorldPos();

        _v2D.push_back(cv::Point2f(kpU.x(), kpU.y()));
        _v3D.push_back(cv::Point3f(Xw.x(), Xw.y(), Xw.z()));
        _index.push_back(i);
//        pFrame->_features[i]->setOutlier();
    }

    _K = pFrame->_camera->_cvK;
    setRansacParameters(500, 3.0f, 0.85);
}

void PnPRansac::setRansacParameters(int iterations, float reprojectionError, double confidence)
{
    _iterationsCount = iterations;
    _reprojectionError = reprojectionError;
    _confidence = confidence;
}

PnPRansac& PnPRansac::setIterations(int iters)
{
    _iterationsCount = iters;
    return *this;
}

PnPRansac& PnPRansac::setReprojectionTh(float reprojection)
{
    _reprojectionError = reprojection;
    return *this;
}

PnPRansac& PnPRansac::setConfidence(double confidence)
{
    _confidence = confidence;
    return *this;
}

bool PnPRansac::compute()
{
    if (_v2D.size() < 10)
        return false;

    cv::Mat r, t, inliers;
    bool bOK = cv::solvePnPRansac(_v3D, _v2D, _K, cv::Mat(), r, t, false, _iterationsCount,
        _reprojectionError, _confidence, inliers);

    if (bOK) {
        cv::Mat Tcw = Converter::toHomogeneous(r, t);
        Mat44 M = Converter::toMatrix4f(Tcw).cast<double>();

        _frame->setPose(SE3(M));

//        for (int i = 0; i < inliers.rows; ++i) {
//            int n = inliers.at<int>(i);
//            const size_t idx = _index[n];
//            _frame->_features[idx]->setInlier();
//        }
    } else {
//        for (size_t i = 0; i < _frame->_N; ++i) {
//            LandmarkPtr pMP = _frame->_features[i]->_point;
//            if (!pMP)
//                continue;
//            _frame->_features[i]->setInlier();
//        }

        cerr << "PnPRansac fail" << endl;
    }

    return bOK;
}
