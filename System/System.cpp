#include "System.h"
#include "Converter.h"
#include "Core/GraphNode.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/Map.h"
#include "Drawer/DenseMapDrawer.h"
#include "Drawer/MapDrawer.h"
#include "Drawer/Viewer.h"
#include "Features/Extractor.h"
#include "IO/Dataset.h"
#include "Mapping.h"
#include "PlaceRecognition/KeyFrameDatabase.h"
#include "PlaceRecognition/LoopClosing.h"
#include "Tracking.h"
#include <iomanip>
#include <iostream>
#include <pangolin/pangolin.h>
#include <thread>
#include <unistd.h>

using namespace std;

System::System(ExtractorPtr pExtractor, VocabularyPtr pVoc, DatasetPtr pDataset, const bool bUseViewer, const bool bUseDenseMap)
    : _vocabulary(pVoc)
    , _viewer(nullptr)
    , _denseMap(nullptr)
    , _reset(false)
{
    _KeyFrameDatabase.reset(new KeyFrameDatabase(_vocabulary));
    _map.reset(new Map());

    _mapDrawer.reset(new MapDrawer(_map));

    if (bUseDenseMap && bUseViewer)
        _denseMap.reset(new DenseMapDrawer(_map));

    _tracker.reset(new Tracking(this, pExtractor, _vocabulary, _mapDrawer,
        _map, _KeyFrameDatabase, pDataset));

    _localMapper.reset(new Mapping(_map));
    _loopCloser.reset(new LoopClosing(_map, _KeyFrameDatabase, _vocabulary));

    if (bUseViewer) {
        _viewer.reset(new Viewer(this, _mapDrawer, _tracker, _map, _denseMap));
        _tracker->setViewer(_viewer);
    }

    _tracker->setLocalMapper(_localMapper);
    _tracker->setLoopClosing(_loopCloser);

    _localMapper->setTracker(_tracker);
    _localMapper->setLoopCloser(_loopCloser);

    _loopCloser->setTracker(_tracker);
    _loopCloser->setLocalMapper(_localMapper);
}

System::~System()
{
    _map->clear();
    cout << "\nMap CLEANED" << endl;
}

SE3 System::trackRGBD(const cv::Mat& im, const cv::Mat& depthmap, const double& timestamp)
{
    {
        unique_lock<mutex> lock(_mutexReset);
        if (_reset) {
            _tracker->reset();
            _reset = false;
        }
    }

    return _tracker->grabImageRGBD(im, depthmap, timestamp);
}

bool System::mapChanged()
{
    static int n = 0;
    int curn = _map->getLastBigChangeIdx();
    if (n < curn) {
        n = curn;
        return true;
    } else
        return false;
}

void System::reset()
{
    unique_lock<mutex> lock(_mutexReset);
    _reset = true;
}

void System::shutdown()
{
    _localMapper->requestFinish();
    _loopCloser->requestFinish();

    if (_viewer) {
        _viewer->requestFinish();

        while (!_viewer->isFinished())
            usleep(5000);
    }

    if (_denseMap) {
        _denseMap->requestFinish();

        while (!_denseMap->isFinished())
            usleep(5000);

        _denseMap->join();
    }

    // Wait until all thread have effectively stopped
    while (!_localMapper->isFinished() || !_loopCloser->isFinished() || _loopCloser->isRunningGBA())
        usleep(5000);

    _localMapper->join();
    _loopCloser->join();

    if (_viewer) {
        pangolin::BindToContext("Map Viewer");
        _viewer->join();
    }
}

void System::saveTrajectoryTUM(const string& filename)
{
    cout << "\nSaving camera trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = _map->getAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    SE3 Two = vpKFs[0]->getPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<KeyFrame*>::iterator lRit = _tracker->_references.begin();
    list<double>::iterator lT = _tracker->_frameTimes.begin();
    list<bool>::iterator lbL = _tracker->_losts.begin();
    for (list<SE3>::iterator lit = _tracker->_relativeFramePoses.begin(),
                             lend = _tracker->_relativeFramePoses.end();
         lit != lend; lit++, lRit++, lT++, lbL++) {
        if (*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        SE3 Trw(Mat44::Identity());

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while (pKF->isBad()) {
            Trw = Trw * pKF->_Tcp;
            pKF = pKF->_node->getParent();
        }

        Trw = Trw * pKF->getPose() * Two;

        SE3 Tcw = (*lit) * Trw;
        Mat33 Rwc = Tcw.rotationMatrix().transpose();
        Quat q(Rwc);
        Vec3 twc = -Rwc * Tcw.translation();

        f << setprecision(6) << *lT << " " << setprecision(9) << twc.x() << " " << twc.y() << " " << twc.z()
          << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    f.close();
    cout << "trajectory saved!" << endl;
}

void System::saveKeyFrameTrajectoryTUM(const string& filename)
{
    cout << "\nSaving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = _map->getAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];

        // pKF->SetPose(pKF->GetPose()*Two);

        if (pKF->isBad())
            continue;

        Mat33 R = pKF->getPoseInverse().rotationMatrix();
        Quat q(R);
        Vec3 t = pKF->getPoseInverse().translation();
        f << setprecision(6) << pKF->_timestamp << setprecision(7) << " " << t.x() << " " << t.y() << " " << t.z()
          << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }

    f.close();
    cout << "trajectory saved!" << endl;
}

void System::saveObservationHistogram(const string& filename)
{
    cout << "\nSaving observations histogram to " << filename << " ..." << endl;

    vector<Landmark*> vpMPs = _map->getAllLandmarks();
    map<int, int> hist;
    for (Landmark* pMP : vpMPs) {
        if (!pMP)
            continue;
        if (pMP->isBad())
            continue;

        hist[pMP->observations()]++;
    }

    ofstream fobs;
    fobs.open(filename.c_str());
    // 5, 10
    // 10 landmarks han sido observadas 5 veces (o desde 5 KFs diferentes)
    for (const auto& [obs, nLMs] : hist)
        fobs << obs << "," << nLMs << endl;

    fobs.close();
    cout << " -Observations histogram saved!" << endl;
}

void System::saveCovisibilityGraph(const string& filename)
{
    cout << "\nSaving Covisibility Graph to " << filename << " ..." << endl;

    ofstream f;
    f.open(filename.c_str());

    vector<KeyFrame*> vpKFs = _map->getAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    for (KeyFrame* pKF : vpKFs) {
        vector<KeyFrame*> vpCovisibleKFs = pKF->_node->getCovisibles();

        f << "[" << pKF->_id << "] ";

        for (KeyFrame* pCovKF : vpCovisibleKFs) {
            f << "-(" << pKF->_node->getWeight(pCovKF) << ")-> [" << pCovKF->_id << "] ";
        }
        f << endl;
    }

    f.close();
    cout << " -Covisibility Graph saved!" << endl;
}

void System::setMeanTime(const double& time)
{
    if (_viewer)
        _viewer->setMeanTime(time);
}

void System::informBigChange()
{
    _map->informNewBigChange();
}
