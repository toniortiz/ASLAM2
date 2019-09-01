#include "Mapping.h"
#include "Core/Feature.h"
#include "Core/GraphNode.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/Map.h"
#include "Core/PinholeCamera.h"
#include "Features/Matcher.h"
#include "PlaceRecognition/LoopClosing.h"
#include "Solver/BundleAdjustment.h"
#include "System/Tracking.h"
#include <mutex>
#include <unistd.h>

using namespace std;

Mapping::Mapping(MapPtr pMap, const bool& start)
    : _resetRequested(false)
    , _finishRequested(false)
    , _finished(true)
    , _map(pMap)
    , _abortBA(false)
    , _stopped(false)
    , _stopRequested(false)
    , _notStop(false)
    , _acceptKFs(true)
{
    if (start)
        _thread = thread(&Mapping::run, this);
}

void Mapping::setLoopCloser(LoopClosingPtr pLoopCloser)
{
    _loopCloser = pLoopCloser;
}

void Mapping::setTracker(TrackingPtr pTracker)
{
    _tracker = pTracker;
}

void Mapping::start()
{
    if (!_thread.joinable())
        _thread = thread(&Mapping::run, this);
}

void Mapping::run()
{
    _finished = false;

    while (1) {
        setAcceptKeyFrames(false);

        if (checkNewKeyFrames()) {
            processNewKeyFrame();

            landmarkCulling();

            if (!checkNewKeyFrames())
                searchInNeighbors();

            _abortBA = false;

            if (!checkNewKeyFrames() && !stopRequested()) {
                if (_map->KeyFramesInMap() > 2)
                    BundleAdjustment::Local(_curKF, &_abortBA, _map);

                KeyFrameCulling();
            }

            _loopCloser->insertKeyFrame(_curKF);
        } else if (stop()) {
            // Safe area to stop
            while (isStopped() && !checkFinish())
                usleep(3000);

            if (checkFinish())
                break;
        }

        resetIfRequested();

        setAcceptKeyFrames(true);

        if (checkFinish())
            break;

        usleep(3000);
    }

    setFinish();
}

void Mapping::insertKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(_mutexQueue);
    _KFsQueue.push_back(pKF);
    _abortBA = true;
}

bool Mapping::checkNewKeyFrames()
{
    unique_lock<mutex> lock(_mutexQueue);
    return (!_KFsQueue.empty());
}

void Mapping::processNewKeyFrame()
{
    {
        unique_lock<mutex> lock(_mutexQueue);
        _curKF = _KFsQueue.front();
        _KFsQueue.pop_front();
    }

    // Compute Bags of Words structures
    _curKF->computeBoW();

    // Associate Landmarks to the new keyframe and update normal and descriptor
    const vector<LandmarkPtr> vpLandmarkMatches = _curKF->getLandmarkMatches();

    for (size_t i = 0; i < vpLandmarkMatches.size(); i++) {
        LandmarkPtr pMP = vpLandmarkMatches[i];
        if (pMP) {
            if (!pMP->isBad()) {
                if (!pMP->isInKeyFrame(_curKF)) {
                    pMP->addObservation(_curKF, i);
                    pMP->updateNormalAndDepth();
                    pMP->computeDistinctiveDescriptors();
                }
                // This can only happen for new points inserted by the Tracking
                else {
                    _recentAddedLandmarks.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    _curKF->_node->updateConnections();

    _map->addKeyFrame(_curKF);
}

void Mapping::landmarkCulling()
{
    list<LandmarkPtr>::iterator lit = _recentAddedLandmarks.begin();
    const int nCurrentKFid = _curKF->_id;

    int nThObs = 3;
    const int cnThObs = nThObs;

    while (lit != _recentAddedLandmarks.end()) {
        LandmarkPtr pMP = *lit;
        if (pMP->isBad()) {
            lit = _recentAddedLandmarks.erase(lit);
        } else if (pMP->getFoundRatio() < 0.25) {
            pMP->setBadFlag();
            lit = _recentAddedLandmarks.erase(lit);
        } else if ((nCurrentKFid - pMP->_firstKFid) >= 2 && pMP->observations() <= cnThObs) {
            pMP->setBadFlag();
            lit = _recentAddedLandmarks.erase(lit);
        } else if ((nCurrentKFid - pMP->_firstKFid) >= 3)
            lit = _recentAddedLandmarks.erase(lit);
        else
            lit++;
    }
}

void Mapping::searchInNeighbors()
{
    // Retrieve neighbor keyframes
    uint nn = 10;
    const vector<KeyFrame*> vpNeighKFs = _curKF->_node->getBestNCovisibles(nn);
    vector<KeyFrame*> vpTargetKFs;
    for (KeyFrame* pKFi : vpNeighKFs) {
        if (pKFi->isBad() || pKFi->_fuseTargetForKF == _curKF->_id)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->_fuseTargetForKF = _curKF->_id;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->_node->getBestNCovisibles(5);
        for (KeyFrame* pKFi2 : vpSecondNeighKFs) {
            if (pKFi2->isBad() || pKFi2->_fuseTargetForKF == _curKF->_id || pKFi2->_id == _curKF->_id)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    Matcher matcher;
    vector<LandmarkPtr> vpLandmarkMatches = _curKF->getLandmarkMatches();
    for (KeyFrame* pKFi : vpTargetKFs)
        matcher.fuse(pKFi, vpLandmarkMatches);

    // Search matches by projection from target KFs in current KF
    vector<LandmarkPtr> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpLandmarkMatches.size());

    for (KeyFrame* pKFi : vpTargetKFs) {
        vector<LandmarkPtr> vpLandmarksKFi = pKFi->getLandmarkMatches();

        for (LandmarkPtr pMP : vpLandmarksKFi) {
            if (!pMP)
                continue;
            if (pMP->isBad() || pMP->_fuseCandidateForKF == _curKF->_id)
                continue;
            pMP->_fuseCandidateForKF = _curKF->_id;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.fuse(_curKF, vpFuseCandidates);

    // Update points
    vpLandmarkMatches = _curKF->getLandmarkMatches();
    for (LandmarkPtr pMP : vpLandmarkMatches) {
        if (pMP) {
            if (!pMP->isBad()) {
                pMP->computeDistinctiveDescriptors();
                pMP->updateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    _curKF->_node->updateConnections();
}

void Mapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the Landmarks it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = _curKF->_node->getCovisibles();

    for (KeyFrame* pKF : vpLocalKeyFrames) {
        if (pKF->_id == 0)
            continue;
        const vector<LandmarkPtr> vpLandmarks = pKF->getLandmarkMatches();

        const int thObs = 3;
        int nRedundantObservations = 0;
        int nMPs = 0;
        for (size_t i = 0, iend = vpLandmarks.size(); i < iend; i++) {
            LandmarkPtr pMP = vpLandmarks[i];
            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;
            if (pKF->getDepth(i) > pKF->_camera->maxDepthTh() || pKF->getDepth(i) < 0)
                continue;

            nMPs++;
            if (pMP->observations() > thObs) {
                const int& scaleLevel = pKF->_features[i]->_level;
                const map<KeyFrame*, size_t> observations = pMP->getObservations();
                int nObs = 0;
                for (auto& [pKFi, idx] : observations) {
                    if (pKFi == pKF)
                        continue;
                    const int& scaleLeveli = pKFi->_features[idx]->_level;

                    if (scaleLeveli <= scaleLevel + 1) {
                        nObs++;
                        if (nObs >= thObs)
                            break;
                    }
                }
                if (nObs >= thObs)
                    nRedundantObservations++;
            }
        }

        if (nRedundantObservations > 0.9 * nMPs)
            pKF->setBadFlag();
    }
}

void Mapping::requestStop()
{
    unique_lock<mutex> lock(_mutexStop);
    _stopRequested = true;
    unique_lock<mutex> lock2(_mutexQueue);
    _abortBA = true;
}

bool Mapping::stop()
{
    unique_lock<mutex> lock(_mutexStop);
    if (_stopRequested && !_notStop) {
        _stopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool Mapping::isStopped()
{
    unique_lock<mutex> lock(_mutexStop);
    return _stopped;
}

bool Mapping::stopRequested()
{
    unique_lock<mutex> lock(_mutexStop);
    return _stopRequested;
}

void Mapping::release()
{
    unique_lock<mutex> lock(_mutexStop);
    unique_lock<mutex> lock2(_mutexFinish);
    if (_finished)
        return;
    _stopped = false;
    _stopRequested = false;
    for (KeyFrame* pKF : _KFsQueue)
        delete pKF;
    _KFsQueue.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool Mapping::acceptKeyFrames()
{
    unique_lock<mutex> lock(_mutexAccept);
    return _acceptKFs;
}

void Mapping::setAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(_mutexAccept);
    _acceptKFs = flag;
}

bool Mapping::setNotStop(bool flag)
{
    unique_lock<mutex> lock(_mutexStop);

    if (flag && _stopped)
        return false;

    _notStop = flag;

    return true;
}

void Mapping::interruptBA()
{
    _abortBA = true;
}

void Mapping::requestReset()
{
    {
        unique_lock<mutex> lock(_mutexReset);
        _resetRequested = true;
    }

    while (1) {
        {
            unique_lock<mutex> lock2(_mutexReset);
            if (!_resetRequested)
                break;
        }
        usleep(3000);
    }
}

void Mapping::resetIfRequested()
{
    unique_lock<mutex> lock(_mutexReset);
    if (_resetRequested) {
        _KFsQueue.clear();
        _recentAddedLandmarks.clear();
        _resetRequested = false;
    }
}

void Mapping::requestFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finishRequested = true;
}

bool Mapping::checkFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finishRequested;
}

void Mapping::setFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finished = true;
    unique_lock<mutex> lock2(_mutexStop);
    _stopped = true;
}

bool Mapping::isFinished()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finished;
}

void Mapping::join()
{
    if (_thread.joinable()) {
        _thread.join();
        cout << "Local Mapping thread JOINED" << endl;
    }
}
