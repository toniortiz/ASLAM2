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

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/Feature_tests/leuven_light/";

struct TestFrame {
    cv::Mat image;
    vector<cv::KeyPoint> kps;
    cv::Mat desc;
};

void match(TestFrame& f1, TestFrame& f2, vector<cv::DMatch>& matches, cv::Ptr<cv::DescriptorMatcher> matcher, float ratio)
{
    vector<vector<cv::DMatch>> matchesKnn;
    set<int> trainIdxs;

    matcher->knnMatch(f1.desc, f2.desc, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < ratio * m2.distance) {
            if (trainIdxs.count(m1.trainIdx) > 0)
                continue;

            trainIdxs.insert(m1.trainIdx);
            matches.push_back(m1);
        }
    }
}

int Ransac(TestFrame& f1, TestFrame& f2, vector<cv::DMatch>& matches, vector<cv::DMatch>& inliers)
{
    vector<uchar> status;
    vector<cv::Point2f> srcPoints, tgtPoints;
    for (const auto& m : matches) {
        srcPoints.push_back(f1.kps[m.queryIdx].pt);
        tgtPoints.push_back(f2.kps[m.trainIdx].pt);
    }
    try {
        cv::Mat H = cv::findHomography(srcPoints, tgtPoints, CV_RANSAC, 6, status, 500, 0.995);
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i])
                inliers.push_back(matches[i]);
        }

    } catch (cv::Exception& ex) {
        cout << ex.what() << endl;
        return 0;
    }
}

void drawMatches(TestFrame& f1, TestFrame& f2, vector<cv::DMatch>& matches, int wait)
{
    cv::Mat out;
    cv::drawMatches(f1.image, f1.kps, f2.image, f2.kps, matches, out);
    cv::imshow("matches", out);
    cv::waitKey(wait);
}

typedef DatasetTUM DatasetT;

int main()
{
    // GFTT - (ORB, BRIEF, SIFT)
    // ORB - (ORB, BRISK, FREAK)
    // FAST - (ORB, BRIEF)
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
        Extractor::BRISK,
        Extractor::BRIEF,
        Extractor::FREAK,
        Extractor::SURF,
        Extractor::SIFT
    };

    for (auto& descriptor : descriptors) {
        Extractor* extractor = new Extractor(Extractor::GFTT, descriptor, Extractor::NORMAL);
        extractor->print(cout);
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(Extractor::mNorm);

        TestFrame base;
        base.image = cv::imread(baseDir + vImageFilenamesRGB[0], cv::IMREAD_COLOR);
        extractor->detectAndCompute(base.image, cv::Mat(), base.kps, base.desc);

        vector<double> rep;
        ofstream fout("./Repeatability/" + extractor->getDetectorName() + "/" + extractor->getDescriptorName() + ".csv");

        for (size_t ni = 1; ni < nImages; ni++) {
            TestFrame ref;
            ref.image = cv::imread(baseDir + vImageFilenamesRGB[ni], cv::IMREAD_COLOR);
            extractor->detectAndCompute(ref.image, cv::Mat(), ref.kps, ref.desc);

            vector<cv::DMatch> matches;
            match(base, ref, matches, matcher, 0.9f);

            vector<cv::DMatch> inliers;
            Ransac(base, ref, matches, inliers);

            rep.push_back(double(inliers.size()) / double(base.kps.size()));

            //            drawMatches(base, ref, inliers, 300);
        }

        for (auto& r : rep)
            fout << r << ",";

        fout.close();
    }

    return 0;
}
