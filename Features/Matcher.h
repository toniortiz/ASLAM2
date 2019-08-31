#ifndef MATCHER_H
#define MATCHER_H

#include "System/Common.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <set>
#include <vector>

class Matcher {
public:
    SMART_POINTER_TYPEDEFS(Matcher);

    Matcher(float nnratio = 0.6, bool checkOri = true);

    // Computes the Hamming distance between two binary descriptors
    static int descriptorDistance(const cv::Mat& a, const cv::Mat& b);

    // Search matches between Frame keypoints and projected Landmarks. Returns number of matches
    // Used to track the local map (Tracking)
    int searchByProjection(FramePtr F, const std::vector<Landmark*>& vpLandmarks, const double th = 3);

    // Project Landmarks tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int searchByProjection(FramePtr prevFrame, FramePtr currFrame, std::vector<cv::DMatch>& vMatches, const double th);
    int knnMatch(const FramePtr F1, FramePtr F2, std::vector<cv::DMatch>& vMatches);

    // Project Landmarks using a SE3 Transformation and search matches.
    // Used in loop detection (Loop Closing)
    int searchByProjection(KeyFrame* pKF, SE3 Tcw, const std::vector<Landmark*>& vpPoints, std::vector<Landmark*>& vpMatched, int th);

    // Used in Loop Detection
    int knnMatch(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<cv::DMatch>& vM21);

    // Used in Tracking for first matching
    int knnMatch(KeyFrame* pKF, FramePtr F, std::vector<Landmark*>& vpLandmarkMatches, std::vector<cv::DMatch>& vMatches);

    // Search matches between Landmarks seen in KF1 and KF2 transforming by a SE3 [R12|t12]
    int searchBySE3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<Landmark*>& vpMatches12, const Mat33& R12, const Vec3& t12, const double th);

    // Project Landmarks into KeyFrame and search for duplicated Landmarks.
    int fuse(KeyFrame* pKF, const std::vector<Landmark*>& vpLandmarks, const double th = 3.0);

    // Project Landmarks into KeyFrame using a given SE3 and search for duplicated Landmarks.
    int fuse(KeyFrame* pKF, SE3 Tcw, const std::vector<Landmark*>& vpPoints, double th, std::vector<Landmark*>& vpReplacePoint);

    static void drawMatches(const Frame& F1, const Frame& F2, const std::vector<cv::DMatch>& m12, const int delay = 1);
    static cv::Mat getImageMatches(const FramePtr F1, const FramePtr F2, const std::vector<cv::DMatch>& m12);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    double radiusByViewingCos(const double& viewCos);

    void computeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3);

    float _NNratio;
    bool _checkOrientation;

    cv::Ptr<cv::DescriptorMatcher> _matcher;
};

#endif // ORBMATCHER_H
