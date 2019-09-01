#include "Matcher.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/PinholeCamera.h"
#include "Extractor.h"
#include <limits.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <stdint-gcc.h>

using namespace std;

const int Matcher::TH_HIGH = 100;
const int Matcher::TH_LOW = 50;
const int Matcher::HISTO_LENGTH = 30;

Matcher::Matcher(float nnratio, bool checkOri)
    : _NNratio(nnratio)
    , _checkOrientation(checkOri)
{
    _matcher = cv::BFMatcher::create(Extractor::_norm);
}

int Matcher::searchByProjection(FramePtr F, const vector<LandmarkPtr>& vpLandmarks, const double th)
{
    int nmatches = 0;

    const bool bFactor = th != 1.0f;

    for (LandmarkPtr pMP : vpLandmarks) {
        if (!pMP->_trackInView)
            continue;

        if (pMP->isBad())
            continue;

        const int& nPredictedLevel = pMP->_trackScaleLevel;

        // The size of the window will depend on the viewing direction
        double r = radiusByViewingCos(pMP->_trackViewCos);

        if (bFactor)
            r *= th;

        const vector<size_t> vIndices = F->getFeaturesInArea(pMP->_trackProjX, pMP->_trackProjY,
            r * F->_scaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

        if (vIndices.empty())
            continue;

        const cv::Mat MPdescriptor = pMP->getDescriptor();

        int bestDist = 256;
        int bestLevel = -1;
        int bestDist2 = 256;
        int bestLevel2 = -1;
        int bestIdx = -1;

        // Get best and second matches with near keypoints
        for (const auto& idx : vIndices) {
            if (F->_features[idx]->_point)
                if (F->_features[idx]->_point->observations() > 0)
                    continue;

            if (F->_features[idx]->_right > 0) {
                const double er = fabs(pMP->_trackProjXR - F->_features[idx]->_right);
                if (er > r * F->_scaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat& d = F->_descriptors.row(idx);

            const int dist = descriptorDistance(MPdescriptor, d);

            if (dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = F->_features[idx]->_level;
                bestIdx = idx;
            } else if (dist < bestDist2) {
                bestLevel2 = F->_features[idx]->_level;
                bestDist2 = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if (bestDist <= TH_HIGH) {
            if (bestLevel == bestLevel2 && bestDist > _NNratio * bestDist2)
                continue;

            F->_features[bestIdx]->_point = pMP;
            nmatches++;
        }
    }

    return nmatches;
}

double Matcher::radiusByViewingCos(const double& viewCos)
{
    if (viewCos > 0.998)
        return 2.5;
    else
        return 4.0;
}

int Matcher::searchByProjection(KeyFrame* pKF, SE3 Tcw, const vector<LandmarkPtr>& vpPoints, vector<LandmarkPtr>& vpMatched, int th)
{
    SE3 Twc = Tcw.inverse();

    // Set of Landmarks already found in the KeyFrame
    set<LandmarkPtr> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(nullptr);

    int nmatches = 0;

    // For each Candidate Landmark Project and Match
    for (LandmarkPtr pMP : vpPoints) {
        // Discard Bad Landmarks and already found
        if (pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        Vec3 Xw = pMP->getWorldPos();

        // Transform into Camera Coords.
        Vec3 Xc = Tcw * Xw;

        // Depth must be positive
        if (Xc.z() < 0.0)
            continue;

        // Project into Image
        Vec2 Xi = pKF->_camera->project(Xc);

        // Point must be inside the image
        if (!pKF->isInImage(Xi.x(), Xi.y()))
            continue;

        // Depth must be inside the scale invariance region of the point
        const double maxDistance = pMP->getMaxDistanceInvariance();
        const double minDistance = pMP->getMinDistanceInvariance();
        Vec3 PO = Xw - Twc.translation();
        const double dist = PO.norm();

        if (dist < minDistance || dist > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Vec3 Pn = pMP->getNormal();

        if (PO.dot(Pn) < 0.5 * dist)
            continue;

        int nPredictedLevel = pMP->predictScale(dist, pKF);

        // Search in a radius
        const double radius = th * pKF->_scaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->getFeaturesInArea(Xi.x(), Xi.y(), radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->getDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for (const auto& idx : vIndices) {
            if (vpMatched[idx])
                continue;

            const int& kpLevel = pKF->_features[idx]->_level;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            const cv::Mat& dKF = pKF->_descriptors.row(idx);

            const int dist = descriptorDistance(dMP, dKF);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_LOW) {
            vpMatched[bestIdx] = pMP;
            nmatches++;
        }
    }

    return nmatches;
}

int Matcher::knnMatch(KeyFrame* pKF1, KeyFrame* pKF2, vector<cv::DMatch>& vM21)
{
    const vector<LandmarkPtr> vpLandmarks1 = pKF1->getLandmarkMatches();
    const vector<LandmarkPtr> vpLandmarks2 = pKF2->getLandmarkMatches();

    vector<LandmarkPtr> vpMatches12 = vector<LandmarkPtr>(vpLandmarks1.size(), nullptr);
    vector<bool> vbMatched2(vpLandmarks2.size(), false);

    int nmatches = 0;

    vector<vector<cv::DMatch>> matchesKnn;
    set<int> trainIdxs;

    _matcher->knnMatch(pKF1->_descriptors, pKF2->_descriptors, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < _NNratio * m2.distance) {
            size_t i1 = static_cast<size_t>(m1.queryIdx);
            size_t i2 = static_cast<size_t>(m1.trainIdx);

            LandmarkPtr pMP1 = vpLandmarks1[i1];
            if (!pMP1)
                continue;
            if (pMP1->isBad())
                continue;

            LandmarkPtr pMP2 = vpLandmarks2[i2];
            if (vbMatched2[i2] || !pMP2)
                continue;
            if (pMP2->isBad())
                continue;

            vpMatches12[i1] = vpLandmarks2[i2];
            vbMatched2[i2] = true;

            // Invert match
            cv::DMatch match(i2, i1, m1.distance);
            vM21.push_back(match);
            nmatches++;
        }
    }

    return nmatches;
}

int Matcher::knnMatch(KeyFrame* pKF, FramePtr F, vector<LandmarkPtr>& vpLandmarkMatches, vector<cv::DMatch>& vMatches)
{
    vector<vector<cv::DMatch>> matchesKnn;
    set<int> trainIdxs;

    _matcher->knnMatch(pKF->_descriptors, F->_descriptors, matchesKnn, 2);

    const vector<LandmarkPtr> vpLandmarksKF1 = pKF->getLandmarkMatches();
    vpLandmarkMatches = vector<LandmarkPtr>(F->_N, nullptr);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < _NNratio * m2.distance) {
            if (trainIdxs.count(m1.trainIdx) > 0)
                continue;

            size_t i1 = static_cast<size_t>(m1.queryIdx);
            size_t i2 = static_cast<size_t>(m1.trainIdx);
            LandmarkPtr pMP = vpLandmarksKF1[i1];

            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;
            if (vpLandmarkMatches[i2])
                continue;

            vpLandmarkMatches[i2] = pMP;

            trainIdxs.insert(m1.trainIdx);
            vMatches.push_back(m1);
        }
    }

    return int(vMatches.size());
}

int Matcher::fuse(KeyFrame* pKF, const vector<LandmarkPtr>& vpLandmarks, const double th)
{
    SE3 Tcw = pKF->getPose();

    const double bf = pKF->_camera->baseLineFx();

    Vec3 Ow = pKF->getPoseInverse().translation();

    int nFused = 0;

    for (LandmarkPtr pMP : vpLandmarks) {
        if (!pMP)
            continue;
        if (pMP->isBad() || pMP->isInKeyFrame(pKF))
            continue;

        Vec3 p3Dw = pMP->getWorldPos();
        Vec3 p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if (p3Dc.z() < 0.0)
            continue;

        const double invz = 1 / p3Dc.z();
        Vec2 xp = pKF->_camera->project(p3Dc);

        // Point must be inside the image
        if (!pKF->isInImage(xp.x(), xp.y()))
            continue;

        const double ur = xp.x() - bf * invz;

        const double maxDistance = pMP->getMaxDistanceInvariance();
        const double minDistance = pMP->getMinDistanceInvariance();
        Vec3 PO = p3Dw - Ow;
        const double dist3D = PO.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Vec3 Pn = pMP->getNormal();

        if (PO.dot(Pn) < 0.5 * dist3D)
            continue;

        int nPredictedLevel = pMP->predictScale(dist3D, pKF);

        // Search in a radius
        const double radius = th * pKF->_scaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->getFeaturesInArea(xp.x(), xp.y(), radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->getDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for (const auto& idx : vIndices) {
            const FeaturePtr kp = pKF->_features[idx];

            const int& kpLevel = kp->_level;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            if (kp->_right >= 0) {
                // Check reprojection error in stereo
                const double kpx = kp->_uXi.x();
                const double kpy = kp->_uXi.y();
                const double kpr = kp->_right;
                const double ex = xp.x() - kpx;
                const double ey = xp.y() - kpy;
                const double er = ur - kpr;
                const double e2 = ex * ex + ey * ey + er * er;

                if (e2 * pKF->_invLevelSigma2[kpLevel] > 7.8)
                    continue;
            } else {
                const double kpx = kp->_uXi.x();
                const double kpy = kp->_uXi.y();
                const double ex = xp.x() - kpx;
                const double ey = xp.y() - kpy;
                const double e2 = ex * ex + ey * ey;

                if (e2 * pKF->_invLevelSigma2[kpLevel] > 5.99)
                    continue;
            }

            const cv::Mat& dKF = pKF->_descriptors.row(idx);

            const int dist = descriptorDistance(dMP, dKF);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a Landmark replace otherwise add new measurement
        if (bestDist <= TH_LOW) {
            LandmarkPtr pMPinKF = pKF->getLandmark(bestIdx);
            if (pMPinKF) {
                if (!pMPinKF->isBad()) {
                    if (pMPinKF->observations() > pMP->observations())
                        pMP->replace(pMPinKF);
                    else
                        pMPinKF->replace(pMP);
                }
            } else {
                pMP->addObservation(pKF, bestIdx);
                pKF->addLandmark(pMP, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int Matcher::fuse(KeyFrame* pKF, SE3 Tcw, const vector<LandmarkPtr>& vpPoints, double th, vector<LandmarkPtr>& vpReplacePoint)
{
    // Set of Landmarks already found in the KeyFrame
    const set<LandmarkPtr> spAlreadyFound = pKF->getLandmarks();

    int nFused = 0;

    const size_t nPoints = vpPoints.size();

    // For each candidate Landmark project and match
    for (size_t iMP = 0; iMP < nPoints; iMP++) {
        LandmarkPtr pMP = vpPoints[iMP];

        // Discard Bad Landmark and already found
        if (pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        Vec3 p3Dw = pMP->getWorldPos();

        // Transform into Camera Coords.
        Vec3 p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if (p3Dc.z() < 0.0)
            continue;

        // Project into Image
        Vec2 xp = pKF->_camera->project(p3Dc);

        // Point must be inside the image
        if (!pKF->isInImage(xp.x(), xp.y()))
            continue;

        // Depth must be inside the scale pyramid of the image
        const double maxDistance = pMP->getMaxDistanceInvariance();
        const double minDistance = pMP->getMinDistanceInvariance();
        Vec3 PO = p3Dw - Tcw.inverse().translation();
        const double dist3D = PO.norm();

        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Vec3 Pn = pMP->getNormal();

        if (PO.dot(Pn) < 0.5 * dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->predictScale(dist3D, pKF);

        // Search in a radius
        const double radius = th * pKF->_scaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->getFeaturesInArea(xp.x(), xp.y(), radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->getDescriptor();

        int bestDist = numeric_limits<int>::max();
        int bestIdx = -1;
        for (const auto& idx : vIndices) {
            const int& kpLevel = pKF->_features[idx]->_level;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            const cv::Mat& dKF = pKF->_descriptors.row(idx);

            int dist = descriptorDistance(dMP, dKF);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a Landmark replace otherwise add new measurement
        if (bestDist <= TH_LOW) {
            LandmarkPtr pMPinKF = pKF->getLandmark(bestIdx);
            if (pMPinKF) {
                if (!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            } else {
                pMP->addObservation(pKF, bestIdx);
                pKF->addLandmark(pMP, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

void Matcher::drawMatches(const Frame& F1, const Frame& F2, const vector<cv::DMatch>& m12, const int delay)
{
    cv::Mat out;

    cv::drawMatches(F1._colorIm, F1._keys, F2._colorIm, F2._keys, m12, out, cv::Scalar::all(-1),
        cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Matches", out);
    cv::waitKey(delay);
}

cv::Mat Matcher::getImageMatches(const FramePtr F1, const FramePtr F2, const std::vector<cv::DMatch>& m12)
{
    cv::Mat out;
    cv::drawMatches(F1->_colorIm, F1->_keys, F2->_colorIm, F2->_keys, m12, out, cv::Scalar::all(-1),
        cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return out;
}

int Matcher::searchBySE3(KeyFrame* pKF1, KeyFrame* pKF2, vector<LandmarkPtr>& vpMatches12, const Mat33& R12, const Vec3& t12, const double th)
{
    // Camera 1 from world
    SE3 T1w = pKF1->getPose();

    // Camera 2 from world
    SE3 T2w = pKF2->getPose();

    // Transformation between cameras
    SE3 T12(R12, t12);
    SE3 T21 = T12.inverse();

    const vector<LandmarkPtr> vpLandmarks1 = pKF1->getLandmarkMatches();
    const size_t N1 = vpLandmarks1.size();

    const vector<LandmarkPtr> vpLandmarks2 = pKF2->getLandmarkMatches();
    const size_t N2 = vpLandmarks2.size();

    vector<bool> vbAlreadyMatched1(N1, false);
    vector<bool> vbAlreadyMatched2(N2, false);

    for (size_t i = 0; i < N1; i++) {
        LandmarkPtr pMP = vpMatches12[i];
        if (pMP) {
            vbAlreadyMatched1[i] = true;
            int idx2 = pMP->getIndexInKeyFrame(pKF2);
            if (idx2 >= 0 && idx2 < N2)
                vbAlreadyMatched2[idx2] = true;
        }
    }

    vector<int> vnMatch1(N1, -1);
    vector<int> vnMatch2(N2, -1);

    // Transform from KF1 to KF2 and search
    for (size_t i1 = 0; i1 < N1; i1++) {
        LandmarkPtr pMP = vpLandmarks1[i1];

        if (!pMP || vbAlreadyMatched1[i1])
            continue;
        if (pMP->isBad())
            continue;

        Vec3 p3Dw = pMP->getWorldPos();
        Vec3 p3Dc1 = T1w * p3Dw;
        Vec3 p3Dc2 = T21 * p3Dc1;

        // Depth must be positive
        if (p3Dc2.z() < 0.0)
            continue;

        Vec2 xp2 = pKF2->_camera->project(p3Dc2);

        // Point must be inside the image
        if (!pKF2->isInImage(xp2.x(), xp2.y()))
            continue;

        const double maxDistance = pMP->getMaxDistanceInvariance();
        const double minDistance = pMP->getMinDistanceInvariance();
        const double dist3D = p3Dc2.norm();

        // Depth must be inside the scale invariance region
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->predictScale(dist3D, pKF2);

        // Search in a radius
        const double radius = th * pKF2->_scaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->getFeaturesInArea(xp2.x(), xp2.y(), radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->getDescriptor();

        int bestDist = numeric_limits<int>::max();
        int bestIdx = -1;
        for (const auto& idx : vIndices) {
            if (pKF2->_features[idx]->_level < nPredictedLevel - 1 || pKF2->_features[idx]->_level > nPredictedLevel)
                continue;

            const cv::Mat& dKF = pKF2->_descriptors.row(idx);

            const int dist = descriptorDistance(dMP, dKF);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH) {
            vnMatch1[i1] = bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for (size_t i2 = 0; i2 < N2; i2++) {
        LandmarkPtr pMP = vpLandmarks2[i2];

        if (!pMP || vbAlreadyMatched2[i2])
            continue;
        if (pMP->isBad())
            continue;

        Vec3 p3Dw = pMP->getWorldPos();
        Vec3 p3Dc2 = T2w * p3Dw;
        Vec3 p3Dc1 = T12 * p3Dc2;

        // Depth must be positive
        if (p3Dc1.z() < 0.0)
            continue;

        Vec2 xp1 = pKF1->_camera->project(p3Dc1);

        // Point must be inside the image
        if (!pKF1->isInImage(xp1.x(), xp1.y()))
            continue;

        const double maxDistance = pMP->getMaxDistanceInvariance();
        const double minDistance = pMP->getMinDistanceInvariance();
        const double dist3D = p3Dc1.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->predictScale(dist3D, pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const double radius = th * pKF1->_scaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->getFeaturesInArea(xp1.x(), xp1.y(), radius);

        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->getDescriptor();

        int bestDist = numeric_limits<int>::max();
        int bestIdx = -1;
        for (const auto& idx : vIndices) {
            if (pKF1->_features[idx]->_level < nPredictedLevel - 1 || pKF1->_features[idx]->_level > nPredictedLevel)
                continue;

            const cv::Mat& dKF = pKF1->_descriptors.row(idx);

            const int dist = descriptorDistance(dMP, dKF);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_HIGH) {
            vnMatch2[i2] = bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for (size_t i1 = 0; i1 < N1; i1++) {
        int idx2 = vnMatch1[i1];

        if (idx2 >= 0) {
            int idx1 = vnMatch2[idx2];
            if (idx1 == i1) {
                vpMatches12[i1] = vpLandmarks2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

int Matcher::searchByProjection(FramePtr prevFrame, FramePtr currFrame, vector<cv::DMatch>& vMatches, const double th)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    SE3 Tc = currFrame->getPose();
    const Mat33 Rcw = Tc.rotationMatrix();
    const Vec3 tcw = Tc.translation();

    const Vec3 twc = -Rcw.transpose() * tcw;

    SE3 Tl = prevFrame->getPose();
    const Mat33 Rlw = Tl.rotationMatrix();
    const Vec3 tlw = Tl.translation();

    const Vec3 tlc = Rlw * twc + tlw;

    const bool bForward = tlc.z() > currFrame->_camera->baseLine();
    const bool bBackward = -tlc.z() > currFrame->_camera->baseLine();

    for (size_t i = 0; i < prevFrame->_N; i++) {
        LandmarkPtr pMP = prevFrame->_features[i]->_point;

        if (pMP) {
            if (prevFrame->_features[i]->isInlier()) {
                Vec3 x3Dw = pMP->getWorldPos();
                Vec3 x3Dc = Tc * x3Dw;

                const double invzc = 1.0 / x3Dc.z();
                if (invzc < 0)
                    continue;

                const Vec2 xp = currFrame->_camera->project(x3Dc);

                if (xp.x() < currFrame->_minX || xp.x() > currFrame->_maxX)
                    continue;
                if (xp.y() < currFrame->_minY || xp.y() > currFrame->_maxY)
                    continue;

                int prevOctave = prevFrame->_keys[i].octave;

                // Search in a window. Size depends on scale
                double radius = th * currFrame->_scaleFactors[prevOctave];

                vector<size_t> vIndices2;

                if (bForward)
                    vIndices2 = currFrame->getFeaturesInArea(xp.x(), xp.y(), radius, prevOctave);
                else if (bBackward)
                    vIndices2 = currFrame->getFeaturesInArea(xp.x(), xp.y(), radius, 0, prevOctave);
                else
                    vIndices2 = currFrame->getFeaturesInArea(xp.x(), xp.y(), radius, prevOctave - 1, prevOctave + 1);

                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->getDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for (const auto& i2 : vIndices2) {
                    if (currFrame->_features[i2]->_point)
                        if (currFrame->_features[i2]->_point->observations() > 0)
                            continue;

                    if (currFrame->_features[i2]->_right > 0) {
                        const double ur = xp.x() - currFrame->_camera->baseLineFx() * invzc;
                        const double er = fabs(ur - currFrame->_features[i2]->_right);
                        if (er > radius)
                            continue;
                    }

                    const cv::Mat& d = currFrame->_descriptors.row(i2);

                    const int dist = descriptorDistance(dMP, d);

                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= TH_HIGH) {
                    currFrame->_features[bestIdx2]->_point = pMP;
                    nmatches++;

                    cv::DMatch match(static_cast<int>(i), bestIdx2, bestDist);
                    vMatches.push_back(match);

                    if (_checkOrientation) {
                        float rot = prevFrame->_features[i]->_angle - currFrame->_features[bestIdx2]->_angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    if (_checkOrientation) {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        computeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++) {
            if (i != ind1 && i != ind2 && i != ind3) {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                    currFrame->_features[rotHist[i][j]]->_point = nullptr;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int Matcher::knnMatch(const FramePtr F1, FramePtr F2, vector<cv::DMatch>& vMatches)
{
    vector<vector<cv::DMatch>> matchesKnn;
    set<int> trainIdxs;

    _matcher->knnMatch(F1->_descriptors, F2->_descriptors, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < _NNratio * m2.distance) {
            if (trainIdxs.count(m1.trainIdx) > 0)
                continue;

            size_t i1 = static_cast<size_t>(m1.queryIdx);
            size_t i2 = static_cast<size_t>(m1.trainIdx);
            LandmarkPtr pMP = F1->_features[i1]->_point;

            if (pMP) {
                if (F1->_features[i1]->isInlier()) {
                    if (F2->_features[i2]->_point) {
                        if (F2->_features[i2]->_point->observations() > 0)
                            continue;
                    }

                    F2->_features[i2]->_point = pMP;
                    trainIdxs.insert(m1.trainIdx);
                    vMatches.push_back(m1);
                }
            }
        }
    }

    return int(vMatches.size());
}

void Matcher::computeThreeMaxima(vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++) {
        const int s = histo[i].size();
        if (s > max1) {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        } else if (s > max2) {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        } else if (s > max3) {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1) {
        ind2 = -1;
        ind3 = -1;
    } else if (max3 < 0.1f * (float)max1) {
        ind3 = -1;
    }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int Matcher::descriptorDistance(const cv::Mat& a, const cv::Mat& b)
{
    const int* pa = a.ptr<int32_t>();
    const int* pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
