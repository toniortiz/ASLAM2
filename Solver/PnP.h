#ifndef PNP_H
#define PNP_H

#include "System/Common.h"
#include <opencv2/features2d.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

class Frame;

class PnP {
public:
    SMART_POINTER_TYPEDEFS(PnP);

    enum eAlgorithm {
        KNEIP = 1, // P3P
        GAO = 2, // P3P
        EPNP = 3, // P6P
    };

    typedef opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem Problem;

public:
    PnP(Frame* pFrame);

    void setRansacParameters(int iterations, double threshold, double probability);

    // Reprojection threshold expresed as pixels tolerance
    PnP& setReprojectionTh(double pixels);
    PnP& setAlgorithm(const eAlgorithm& algorithm);
    PnP& setIterations(int its);
    PnP& refine(bool b);

    bool compute();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    Frame* _frame;

    std::vector<Vec3> _bearings;
    std::vector<Vec3> _landmarks;

    std::vector<size_t> _index;

    // Calibration
    double _focalLength;

    // Ransac parameters
    int _iterations;
    double _threshold;
    double _probability;

    bool _refinement;

    // Guess
    Vec3 _twc;
    Mat33 _Rwc;

    Problem::Algorithm _algorithm;
};

#endif // PNP_H
