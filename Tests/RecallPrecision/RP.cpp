#include "Core/Frame.h"
#include "Core/MapPoint.h"
#include "Features/Extractor.h"
#include "IO/Dataset.h"
#include "IO/DatasetTUM.h"
#include "System/System.h"
#include "gnuplot-cpp/gnuplot_i.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pcl/common/time.h>

using namespace std;

//const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/TUM/rgbd_dataset_freiburg1_room/";
//const string vocDir = "./Vocabulary/voc_TUM_FAST_BRIEF.yml.gz";

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/Feature_tests/wall_viewpoint/";

struct TestFrame {
    cv::Mat image;
    vector<cv::KeyPoint> kps;
    cv::Mat desc;
};

bool findHomography(const TestFrame& f1, const TestFrame& f2, const vector<cv::DMatch>& vMatches12, cv::Mat& H, vector<uchar>& vRansacStatus)
{
    vector<cv::Point2f> vSourcePoints, vTargetPoints;
    vSourcePoints.reserve(vMatches12.size());
    vTargetPoints.reserve(vMatches12.size());

    for (const auto& m : vMatches12) {
        vSourcePoints.push_back(f1.kps[m.queryIdx].pt);
        vTargetPoints.push_back(f2.kps[m.trainIdx].pt);
    }

    try {
        H = cv::findHomography(vSourcePoints, vTargetPoints, CV_RANSAC, 4, vRansacStatus, 500, 0.995);
        //        Mat Fundamental = findFundamentalMat(p1, p2, RansacStatus, FM_RANSAC);
        return true;
    } catch (cv::Exception& ex) {
        cout << ex.what() << endl;
        return false;
    }
}

cv::Point2f applyHomography(const cv::Point2f& pt, const cv::Mat& H)
{
    if (!H.empty()) {
        double d = H.at<double>(6) * pt.x + H.at<double>(7) * pt.y + H.at<double>(8);

        cv::Point2f newPt;
        newPt.x = (H.at<double>(0) * pt.x + H.at<double>(1) * pt.y + H.at<double>(2)) / d;
        newPt.y = (H.at<double>(3) * pt.x + H.at<double>(4) * pt.y + H.at<double>(5)) / d;

        return newPt;
    } else {
        return pt;
    }
}

void drawMatches(TestFrame& f1, TestFrame& f2, vector<cv::DMatch>& matches, int wait)
{
    cv::Mat out;
    cv::drawMatches(f1.image, f1.kps, f2.image, f2.kps, matches, out);
    cv::imshow("matches", out);
    cv::waitKey(wait);
}

vector<pair<double, double>> RP(TestFrame& f1, TestFrame& f2, vector<cv::DMatch>& matches, cv::Ptr<cv::DescriptorMatcher> matcher)
{
    vector<vector<cv::DMatch>> matchesKnn;
    vector<bool> isMatch;

    matcher->knnMatch(f1.desc, f2.desc, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < 0.9f * m2.distance) {
            isMatch.push_back(true);
            matches.push_back(m1);
        } else {
            isMatch.push_back(false);
            matches.push_back(m1);
        }
    }

    vector<pair<double, double>> vDataRP;

    vector<uchar> vRansacStatus;
    cv::Mat H(3, 3, CV_64F);
    if (!findHomography(f1, f2, matches, H, vRansacStatus)) {
        vDataRP.push_back({ 0, 0 });
        return vDataRP;
    }

    // Calculate maximum Euclidean distance between matches according to Homography
    vector<double> vDistances;
    double maxEuclideanDist = 0;
    for (size_t i = 0; i < matches.size(); i++) {
        cv::Point2f src = f1.kps[matches[i].queryIdx].pt;
        cv::Point2f tgt = f2.kps[matches[i].trainIdx].pt;

        cv::Point2f srcT = applyHomography(src, H);
        double dist = cv::norm(cv::Vec2f(srcT.x, srcT.y) - cv::Vec2f(tgt.x, tgt.y));
        vDistances.push_back(dist);

        if (dist > maxEuclideanDist)
            maxEuclideanDist = dist;
    }

    // Varying the threshold between what are regarded as a correct and false positives,
    // different values of recall an precision can be calculated
    Eigen::ArrayXd threshList = Eigen::ArrayXd::LinSpaced(300, 0.0, maxEuclideanDist + 1.0);
    for (int i = 0; i < threshList.size(); ++i) {
        int positives = 0;
        int negatives = 0;
        int truePositives = 0;
        int falsePositives = 0;

        for (size_t j = 0; j < matches.size(); j++) {
            if (isMatch[j]) {
                positives++;
                if (vDistances[j] < threshList(i, 0))
                    truePositives++;
            } else {
                negatives++;
                if (vDistances[j] < threshList(i, 0))
                    falsePositives++;
            }
        }

        double recall = double(truePositives) / double(positives);
        double precision = double(falsePositives) / double(negatives);
        vDataRP.push_back({ recall, precision });
    }

    return vDataRP;
}

typedef DatasetTUM DatasetT;

int main()
{
    vector<string> vImageFilenamesRGB;
    for (int i = 0; i < 6; ++i) {
        stringstream ss;
        ss << "img" << i + 1 << ".ppm";
        vImageFilenamesRGB.push_back(ss.str());
    }
    size_t nImages = vImageFilenamesRGB.size();
    cout << "Start processing sequence: " << baseDir
         << "\nImages in the sequence: " << nImages << endl
         << endl;

        vector<Extractor::eAlgorithm> descriptors{
            Extractor::ORB,
            //        Extractor::BRISK,
            Extractor::BRIEF
                //        Extractor::FREAK,
                };

    for (auto& descriptor : descriptors) {
        Extractor* extractor = new Extractor(Extractor::FAST, descriptor, Extractor::NORMAL);
        extractor->print(cout);
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(Extractor::mNorm);

        TestFrame base;
        base.image = cv::imread(baseDir + vImageFilenamesRGB[0], cv::IMREAD_COLOR);
        extractor->detectAndCompute(base.image, cv::Mat(), base.kps, base.desc);

        TestFrame ref;
        ref.image = cv::imread(baseDir + vImageFilenamesRGB[3], cv::IMREAD_COLOR);
        extractor->detectAndCompute(ref.image, cv::Mat(), ref.kps, ref.desc);

        vector<cv::DMatch> matches;
        vector<pair<double, double>> rpdata = RP(base, ref, matches, matcher);

        //            drawMatches(base, ref, inliers, 300);

        ofstream fout(extractor->getDetectorName() + "-" + extractor->getDescriptorName() + ".csv");
        for (auto& [r, p] : rpdata)
            fout << r << "," << p << endl;
        fout.close();
    }

    //    DatasetT* dataset = new DatasetT();
    //    dataset->open(baseDir);
    //    dataset->print(cout);

    //    Extractor* extractor = new Extractor(Extractor::FAST, Extractor::BRIEF, Extractor::ADAPTIVE);
    //    extractor->print(cout);

    //    cout << "Loading Vocabulary..." << flush;
    //    DBoW3::Vocabulary* vocab = new DBoW3::Vocabulary(vocDir);
    //    cout << " done!" << endl;

    //    System SLAM(extractor, vocab, dataset, true);

    //    vector<double> vTimesTrack;
    //    vTimesTrack.reserve(dataset->size());
    //    cv::TickMeter tm;

    //    for (size_t ni = 0; ni < dataset->size(); ni++) {
    //        auto [imgs, timestamp] = dataset->getRGBDFrame(ni);

    //        pcl::StopWatch s1;
    //        tm.start();
    //        SLAM.trackRGBD(imgs.first, imgs.second, timestamp);
    //        tm.stop();

    //        vTimesTrack.push_back(s1.getTime());
    //        SLAM.setMeanTime(tm.getTimeSec() / tm.getCounter());
    //    }

    //    cout << "\n\nMean tracking time: " << tm.getTimeSec() / tm.getCounter() << endl;

    //    Gnuplot g1("lines");
    //    g1.set_title("Processing time").plot_x(vTimesTrack);
    //    g1 << " replot " + to_string(tm.getTimeMilli() / tm.getCounter());

    //    cout << "\nPress enter to close" << endl;
    //    getchar();

    //    // Stop all threads
    //    SLAM.shutdown();

    //    SLAM.saveTrajectoryTUM("CameraTrajectory.txt");
    //    SLAM.saveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    //    delete extractor;
    //    delete vocab;
    //    delete dataset;

    return 0;
}

