#ifndef CONVERTER_H
#define CONVERTER_H

#include "Common.h"
#include <Eigen/Dense>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <opencv2/core/core.hpp>

class Converter {
public:
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat& Descriptors);

    static g2o::SE3Quat toSE3Quat(const cv::Mat& cvT);
    static g2o::SE3Quat toSE3Quat(const g2o::Sim3& gSim3);

    static cv::Mat toCvMat(const g2o::SE3Quat& SE3);
    static cv::Mat toCvMat(const g2o::Sim3& Sim3);
    static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4>& m);
    static cv::Mat toCvMat(const Eigen::Matrix3d& m);
    static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1>& m);
    static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3>& R, const Eigen::Matrix<double, 3, 1>& t);

    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat& cvVector);
    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f& cvPoint);
    static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat& cvMat3);

    static std::vector<double> toQuaternion(const Mat33& M);

    template <typename type, int rows, int cols>
    static cv::Mat toMat(const Eigen::Matrix<type, rows, cols>& matrix)
    {
        cv::Mat mat(rows, cols, CV_32F);

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                mat.at<float>(i, j) = matrix(i, j);

        return mat;
    }

    static cv::Mat toCvSE3(const cv::Mat& R, const cv::Mat t);

    static cv::Mat toHomogeneous(const cv::Mat& r, const cv::Mat& t);

    static Eigen::Matrix4f toMatrix4f(const cv::Mat& cvMat);
};

#endif // CONVERTER_H
