#include "Features/Extractor.h"
#include "IO/Dataset.h"
#include "IO/DatasetAICL.h"
#include "IO/DatasetCORBS.h"
#include "IO/DatasetICL.h"
#include "IO/DatasetMicrosoft.h"
#include "IO/DatasetTUM.h"
#include "System/System.h"
#include <algorithm>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/AICL/livingroom1/";
const string vocDir = "./Vocabulary/voc_TUM_ORB_ORB.yml.gz";

int main()
{
    Dataset::Ptr dataset(new DatasetAICL());
    dataset->open(baseDir);
    dataset->print(cout);

    Extractor::Ptr extractor(new Extractor(Extractor::ORB, Extractor::ORB, Extractor::ADAPTIVE));
    extractor->_gridResolution = 3;
    extractor->print(cout);

    cout << "Loading Vocabulary..." << flush;
    VocabularyPtr vocab(new Vocabulary(vocDir));
    cout << " done!" << endl;

    System SLAM(extractor, vocab, dataset, true, true);

    cv::TickMeter tm;
    for (size_t ni = 0; ni < dataset->size(); ni++) {
        auto [imgs, timestamp] = dataset->getData(ni);

        tm.start();
        SLAM.trackRGBD(imgs.first, imgs.second, timestamp);
        tm.stop();

        SLAM.setMeanTime(tm.getTimeSec() / tm.getCounter());
    }
    SLAM.informBigChange();

    cout << "\n\nMean tracking time: " << tm.getTimeSec() / tm.getCounter() << endl;
    cout << "\nPress enter to close" << endl;
    getchar();

    // Stop all threads
    SLAM.shutdown();

    SLAM.saveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.saveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}
