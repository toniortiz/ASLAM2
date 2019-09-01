#ifndef KABSCH_H
#define KABSCH_H

#include "System/Macros.h"
#include <Eigen/Core>

class Kabsch {
public:
    SMART_POINTER_TYPEDEFS(Kabsch);

public:
    Kabsch();

    Eigen::Matrix4f Compute(const Eigen::MatrixXf& setA, const Eigen::MatrixXf& setB);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Eigen::Matrix4f mTransformation;
};
#endif // KABSCH_H
