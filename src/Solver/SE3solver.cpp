#include "SE3solver.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "System/Converter.h"
#include "System/Random.h"
#include <pcl/common/transformation_from_correspondences.h>

using namespace std;

SE3solver::SE3solver(FramePtr pF1, FramePtr pF2, const vector<cv::DMatch>& vMatches12)
    : _vC1(pF1->_features)
    , _vC2(pF2->_features)
    , _frame1(pF1)
    , _frame2(pF2)
{
    _matchesToUse.reserve(vMatches12.size());

    for (const auto& m : vMatches12) {
        const Vec3& source = _vC1[m.queryIdx]->_Xc;
        const Vec3& target = _vC2[m.trainIdx]->_Xc;

        if (isnan(source.z()) || isnan(target.z()))
            continue;
        if (source.z() <= 0 || target.z() <= 0)
            continue;

        _matchesToUse.push_back(m);
        //        pF2->_features[m.trainIdx]->setOutlier();
    }

    setRansacParameters(200, 20, 3.0f, 4);
}

SE3solver::SE3solver(KeyFrame* pKF1, KeyFrame* pKF2, const vector<cv::DMatch>& vMatches12)
    : _vC1(pKF1->_features)
    , _vC2(pKF2->_features)
    , _frame1(nullptr)
    , _frame2(nullptr)
{
    _matchesToUse.reserve(vMatches12.size());

    for (const auto& m : vMatches12) {
        const Vec3& source = _vC1[m.queryIdx]->_Xc;
        const Vec3& target = _vC2[m.trainIdx]->_Xc;

        if (isnan(source.z()) || isnan(target.z()))
            continue;
        if (source.z() <= 0 || target.z() <= 0)
            continue;

        _matchesToUse.push_back(m);
    }

    setRansacParameters(200, 20, 3.0f, 4);
}

void SE3solver::setRansacParameters(int iters, uint minInlierTh, float maxMahalanobisDist, uint sampleSize)
{
    _iterations = iters;
    _minInlierTh = minInlierTh;
    _maxMahalanobisDistance = maxMahalanobisDist;
    _sampleSize = sampleSize;
}

SE3solver& SE3solver::setIterations(int iters)
{
    _iterations = iters;
    return *this;
}

SE3solver& SE3solver::setMinInlierThreshold(uint th)
{
    _minInlierTh = th;
    return *this;
}

SE3solver& SE3solver::setMaxMahalanobisDistance(float md)
{
    _maxMahalanobisDistance = md;
    return *this;
}

SE3solver& SE3solver::setSampleSize(uint ss)
{
    _sampleSize = ss;
    return *this;
}

bool SE3solver::compute()
{
    if (_matchesToUse.size() < _minInlierTh)
        return false;

    _inliers.clear();
    _rmse = 1e6;
    _T = SE3(Mat44::Identity());
    Eigen::Matrix4f T21 = Eigen::Matrix4f::Identity();

    bool validTf = false;
    int realIters = 0;
    int validIters = 0;
    double inlierError;

    sort(_matchesToUse.begin(), _matchesToUse.end());

    for (int n = 0; (n < _iterations && _matchesToUse.size() >= _sampleSize); n++) {
        double refinedError = 1e6;
        vector<cv::DMatch> vRefinedMatches;
        vector<cv::DMatch> vInlierMatches = sampleMatches();
        Eigen::Matrix4f refinedTransformation = Eigen::Matrix4f::Identity();
        realIters++;

        for (int refinements = 1; refinements < 20; refinements++) {
            Eigen::Matrix4f transformation = computeHypothesis(vInlierMatches, validTf);

            if (!validTf && transformation != transformation)
                break;

            inlierError = computeError(transformation, vInlierMatches);

            if (vInlierMatches.size() < _minInlierTh || inlierError > _maxMahalanobisDistance)
                break;

            if (vInlierMatches.size() >= vRefinedMatches.size() && inlierError <= refinedError) {
                size_t prevNumInliers = vRefinedMatches.size();
                assert(inlierError >= 0);
                refinedTransformation = transformation;
                vRefinedMatches = vInlierMatches;
                refinedError = inlierError;

                if (vInlierMatches.size() == prevNumInliers)
                    break;
            } else {
                break;
            }
        }

        if (vRefinedMatches.size() > 0) {
            validIters++;

            if (refinedError <= _rmse && vRefinedMatches.size() >= _inliers.size()
                && vRefinedMatches.size() >= _minInlierTh) {
                _rmse = refinedError;
                T21 = refinedTransformation;
                _T = SE3(T21.cast<double>());
                _inliers.assign(vRefinedMatches.begin(), vRefinedMatches.end());

                if (vRefinedMatches.size() > _matchesToUse.size() * 0.5)
                    n += 10;
                if (vRefinedMatches.size() > _matchesToUse.size() * 0.75)
                    n += 10;
                if (vRefinedMatches.size() > _matchesToUse.size() * 0.8)
                    break;
            }
        }
    }

    if (validIters == 0) {
        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
        vector<cv::DMatch> vInlierMatches;
        inlierError = computeError(transformation, vInlierMatches);

        if (vInlierMatches.size() > _minInlierTh && inlierError < _maxMahalanobisDistance) {
            assert(inlierError >= 0);
            T21 = transformation;
            _T = SE3(T21.cast<double>());
            _inliers.assign(vInlierMatches.begin(), vInlierMatches.end());
            _rmse += inlierError;
            validIters++;
        }
    }

    bool status = _inliers.size() >= _minInlierTh;

    if (_frame1) {
        if (status) {
            //            for (const auto& m : _inliers)
            //                _frame2->_features[m.trainIdx]->setInlier();

            _frame2->setPose(_T * _frame1->getPose());
        } else {
            //            for (const auto& m : _matchesToUse)
            //                _frame2->_features[m.trainIdx]->setInlier();
        }
    }

    return status;
}

vector<cv::DMatch> SE3solver::sampleMatches()
{
    set<vector<cv::DMatch>::size_type> sSampledIds;
    int safetyNet = 0;

    while (sSampledIds.size() < _sampleSize && _matchesToUse.size() >= _sampleSize) {
        int id1 = Random::randomInt(0, _matchesToUse.size() - 1);
        int id2 = Random::randomInt(0, _matchesToUse.size() - 1);

        if (id1 > id2)
            id1 = id2;

        sSampledIds.insert(id1);

        if (++safetyNet > 10000)
            break;
    }

    vector<cv::DMatch> vSampledMatches;
    vSampledMatches.reserve(sSampledIds.size());
    for (const auto& id : sSampledIds)
        vSampledMatches.push_back(_matchesToUse[id]);

    return vSampledMatches;
}

Eigen::Matrix4f SE3solver::computeHypothesis(const vector<cv::DMatch>& vMatches, bool& valid)
{
    pcl::TransformationFromCorrespondences tfc;
    valid = true;
    float weight = 1.0;

    for (const auto& m : vMatches) {
        const Vec3& from = _vC1[m.queryIdx]->_Xc;
        const Vec3& to = _vC2[m.trainIdx]->_Xc;

        if (isnan(from.z()) || isnan(to.z()))
            continue;

        weight = 1.0f / (from.z() * to.z());
        tfc.add(Eigen::Vector3f(from.x(), from.y(), from.z()), Eigen::Vector3f(to.x(), to.y(), to.z()), weight);
    }

    return tfc.getTransformation().matrix();
}

double SE3solver::computeError(const Eigen::Matrix4f& transformation4f, vector<cv::DMatch>& vInlierMatches)
{
    vInlierMatches.clear();
    vInlierMatches.reserve(_matchesToUse.size());

    double meanError = 0.0;
    Eigen::Matrix4d transformation4d = transformation4f.cast<double>();

    for (const auto& m : _matchesToUse) {
        const Vec3& origin = _vC1[m.queryIdx]->_Xc;
        const Vec3& target = _vC2[m.trainIdx]->_Xc;

        if (origin.z() == 0.0 || target.x() == 0.0)
            continue;

        double mahalDist = errorFunction2(Eigen::Vector4f(origin.x(), origin.y(), origin.z(), 1.0), Eigen::Vector4f(target.x(), target.y(), target.z(), 1.0), transformation4d);

        if (mahalDist > (_maxMahalanobisDistance * _maxMahalanobisDistance))
            continue;
        if (!(mahalDist >= 0.0))
            continue;

        meanError += mahalDist;
        vInlierMatches.push_back(m);
    }

    if (vInlierMatches.size() < 3) {
        meanError = 1e9;
    } else {
        meanError /= vInlierMatches.size();
        meanError = sqrt(meanError);
    }

    return meanError;
}

double SE3solver::errorFunction2(const Eigen::Vector4f& x1, const Eigen::Vector4f& x2, const Eigen::Matrix4d& transformation)
{
    static const double cam_angle_x = 58.0 / 180.0 * M_PI;
    static const double cam_angle_y = 45.0 / 180.0 * M_PI;
    static const double cam_resol_x = 640;
    static const double cam_resol_y = 480;
    static const double raster_stddev_x = 3 * tan(cam_angle_x / cam_resol_x);
    static const double raster_stddev_y = 3 * tan(cam_angle_y / cam_resol_y);
    static const double raster_cov_x = raster_stddev_x * raster_stddev_x;
    static const double raster_cov_y = raster_stddev_y * raster_stddev_y;
    static const bool use_error_shortcut = true;

    bool nan1 = std::isnan(x1(2));
    bool nan2 = std::isnan(x2(2));
    if (nan1 || nan2)
        return std::numeric_limits<double>::max();

    Eigen::Vector4d x_1 = x1.cast<double>();
    Eigen::Vector4d x_2 = x2.cast<double>();

    Eigen::Matrix4d tf_12 = transformation;
    Eigen::Vector3d mu_1 = x_1.head<3>();
    Eigen::Vector3d mu_2 = x_2.head<3>();
    Eigen::Vector3d mu_1_in_frame_2 = (tf_12 * x_1).head<3>(); // μ₁⁽²⁾  = T₁₂ μ₁⁽¹⁾

    if (use_error_shortcut) {
        double delta_sq_norm = (mu_1_in_frame_2 - mu_2).squaredNorm();
        double sigma_max_1 = std::max(raster_cov_x, depthCovariance(mu_1(2)));
        double sigma_max_2 = std::max(raster_cov_x, depthCovariance(mu_2(2)));

        if (delta_sq_norm > 2.0 * (sigma_max_1 + sigma_max_2))
            return std::numeric_limits<double>::max();
    }

    Eigen::Matrix3d rotation_mat = tf_12.block(0, 0, 3, 3);

    //Point 1
    Eigen::Matrix3d cov1 = Eigen::Matrix3d::Zero();
    cov1(0, 0) = raster_cov_x * mu_1(2);
    cov1(1, 1) = raster_cov_y * mu_1(2);
    cov1(2, 2) = depthCovariance(mu_1(2));

    //Point2
    Eigen::Matrix3d cov2 = Eigen::Matrix3d::Zero();
    cov2(0, 0) = raster_cov_x * mu_2(2);
    cov2(1, 1) = raster_cov_y * mu_2(2);
    cov2(2, 2) = depthCovariance(mu_2(2));

    Eigen::Matrix3d cov1_in_frame_2 = rotation_mat.transpose() * cov1 * rotation_mat;

    // Δμ⁽²⁾ =  μ₁⁽²⁾ - μ₂⁽²⁾
    Eigen::Vector3d delta_mu_in_frame_2 = mu_1_in_frame_2 - mu_2;
    if (std::isnan(delta_mu_in_frame_2(2)))
        return std::numeric_limits<double>::max();

    // Σc = (Σ₁ + Σ₂)
    Eigen::Matrix3d cov_mat_sum_in_frame_2 = cov1_in_frame_2 + cov2;
    //ΔμT Σc⁻¹Δμ
    double sqrd_mahalanobis_distance = delta_mu_in_frame_2.transpose() * cov_mat_sum_in_frame_2.llt().solve(delta_mu_in_frame_2);

    if (!(sqrd_mahalanobis_distance >= 0.0))
        return std::numeric_limits<double>::max();

    return sqrd_mahalanobis_distance;
}

double SE3solver::depthCovariance(double depth)
{
    static double stddev = depthStdDev(depth);
    static double cov = stddev * stddev;
    return cov;
}

double SE3solver::depthStdDev(double depth)
{
    // From Khoselham and Elberink
    // Factor c for the standard deviation of depth measurements: sigma_Z = c * depth * depth.
    // Khoshelham 2012 (0.001425) seems to be a bit overconfident."
    static double depth_std_dev = 0.01;

    return depth_std_dev * depth * depth;
}
