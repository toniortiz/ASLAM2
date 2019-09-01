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

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/TUM/rgbd_dataset_freiburg1_room/";
const string vocDir = "./Vocabulary/voc_TUM_FAST_BRIEF.yml.gz";

typedef DatasetTUM DatasetT;

int main()
{
    // Speed test
    DatasetT* dataset = new DatasetT();
    dataset->open(baseDir);
    //    dataset->print(cout);

    vector<Extractor::eAlgorithm> descriptors{
        Extractor::ORB,
        Extractor::BRISK,
        Extractor::BRIEF,
        Extractor::FREAK,
        Extractor::LATCH,
        Extractor::SURF,
        Extractor::SIFT
    };

    for (auto& descriptor : descriptors) {
        Extractor* extractor = new Extractor(Extractor::STAR, descriptor, Extractor::NORMAL);
        extractor->print(cout);

        vector<pair<size_t, double>> stats;

        for (size_t ni = 0; ni < dataset->size() / 5; ni++) {
            auto [imgs, timestamp] = dataset->getRGBDFrame(ni);
            const cv::Mat& color = imgs.first;

            vector<cv::KeyPoint> kps;
            cv::Mat desc;
            pcl::StopWatch s;

            extractor->detect(color, kps);
            extractor->compute(color, kps, desc);

            double time = s.getTime();
            stats.push_back({ kps.size(), time });
        }

        double msAcum = 0;
        double timeAcum = 0;
        for (auto& stat : stats) {

            const size_t& features = stat.first;
            const double& time = stat.second;

            double msPerFeature = time / features;

            msAcum += msPerFeature;
            timeAcum += time;
        }
        //cout << "\ntime: " << timeAcum / stats.size() << endl;
        //cout << "ms per feature: " << msAcum / stats.size() << endl;
        cout << "us per feature: " << round((msAcum / stats.size()) * 1000) << endl
             << endl;
    }

    return 0;
}
