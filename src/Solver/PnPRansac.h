#ifndef PNPRANSAC_H
#define PNPRANSAC_H

#include "System/Common.h"
#include <opencv2/features2d.hpp>
#include <vector>

class PnPRansac {
public:
    SMART_POINTER_TYPEDEFS(PnPRansac);

public:
    PnPRansac(FramePtr pFrame);

    void setRansacParameters(int iterations, float reprojectionError, double confidence);

    PnPRansac& setIterations(int iters);
    PnPRansac& setReprojectionTh(float reprojection);
    PnPRansac& setConfidence(double confidence);

    bool compute();

protected:
    FramePtr _frame;

    std::vector<cv::Point2f> _v2D;
    std::vector<cv::Point3f> _v3D;
    std::vector<size_t> _index;

    // Calibration
    cv::Mat _K;

    // Ransac parameters
    int _iterationsCount;
    float _reprojectionError;
    double _confidence;
};

#endif // PNPRANSAC_H
