#include "LoopClosing.h"
#include "Core/GraphNode.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/Map.h"
#include "Features/Matcher.h"
#include "KeyFrameDatabase.h"
#include "Solver/BundleAdjustment.h"
#include "Solver/Icp.h"
#include "Solver/PoseGraph.h"
#include "System/Converter.h"
#include "System/Mapping.h"
#include <mutex>
#include <thread>
#include <unistd.h>

using namespace std;

LoopClosing::LoopClosing(MapPtr pMap, KeyFrameDatabasePtr pDB, VocabularyPtr pVoc, const bool& start)
    : _resetRequested(false)
    , _finishRequested(false)
    , _finished(true)
    , _map(pMap)
    , _KFDB(pDB)
    , _vocabulary(pVoc)
    , _matchedKF(nullptr)
    , _prevLoopKFid(0)
    , _runningGBA(false)
    , _finishedGBA(true)
    , _stopGBA(false)
    , _threadGBA(nullptr)
    , _fullBAIdx(0)
{
    _covisibilityConsistencyTh = 3;

    if (start)
        _thread = thread(&LoopClosing::run, this);
}

void LoopClosing::setTracker(TrackingPtr pTracker) { _tracker = pTracker; }

void LoopClosing::setLocalMapper(MappingPtr pLocalMapper) { _localMapper = pLocalMapper; }

void LoopClosing::start()
{
    if (!_thread.joinable())
        _thread = thread(&LoopClosing::run, this);
}

void LoopClosing::run()
{
    _finished = false;

    while (1) {
        // Check if there are keyframes in the queue
        if (checkNewKeyFrames()) {

            // Detect loop candidates and check covisibility consistency
            if (detectLoop()) {

                // Compute SE3 transformation [R|t]
                if (computeSE3()) {

                    // Perform loop fusion and pose graph optimization
                    correctLoop();
                }
            }
        }

        resetIfRequested();

        if (checkFinish())
            break;

        usleep(5000);
    }

    setFinish();
}

void LoopClosing::insertKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(_mutexQueue);
    if (pKF->_id != 0)
        _KFsQueue.push_back(pKF);
}

bool LoopClosing::checkNewKeyFrames()
{
    unique_lock<mutex> lock(_mutexQueue);
    return (!_KFsQueue.empty());
}

bool LoopClosing::detectLoop()
{
    {
        unique_lock<mutex> lock(_mutexQueue);
        _curKF = _KFsQueue.front();
        _KFsQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        _curKF->setNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if (_curKF->_id < _prevLoopKFid + 10) {
        _KFDB->add(_curKF);
        _curKF->setErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<KeyFrame*> vpConnectedKeyFrames = _curKF->_node->getCovisibles();
    const DBoW3::BowVector& CurrentBowVec = _curKF->_bowVec;
    float minScore = 1;
    for (KeyFrame* pKF : vpConnectedKeyFrames) {
        if (pKF->isBad())
            continue;
        const DBoW3::BowVector& BowVec = pKF->_bowVec;

        float score = static_cast<float>(_vocabulary->score(CurrentBowVec, BowVec));

        if (score < minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    vector<KeyFrame*> vpCandidateKFs = _KFDB->detectLoopCandidates(_curKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if (vpCandidateKFs.empty()) {
        _KFDB->add(_curKF);
        _consistentGroups.clear();
        _curKF->setErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    _enoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(_consistentGroups.size(), false);
    for (KeyFrame* pCandidateKF : vpCandidateKFs) {
        set<KeyFrame*> spCandidateGroup = pCandidateKF->_node->getConnectedKFs();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for (size_t iG = 0, iendG = _consistentGroups.size(); iG < iendG; iG++) {
            set<KeyFrame*> sPreviousGroup = _consistentGroups[iG].first;

            bool bConsistent = false;
            for (set<KeyFrame*>::iterator sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++) {
                if (sPreviousGroup.count(*sit)) {
                    bConsistent = true;
                    bConsistentForSomeGroup = true;
                    break;
                }
            }

            if (bConsistent) {
                int nPreviousConsistency = _consistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if (!vbConsistentGroup[iG]) {
                    ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
                }
                if (nCurrentConsistency >= _covisibilityConsistencyTh && !bEnoughConsistent) {
                    _enoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent = true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if (!bConsistentForSomeGroup) {
            ConsistentGroup cg = make_pair(spCandidateGroup, 0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    _consistentGroups = vCurrentConsistentGroups;

    // Add Current Keyframe to database
    _KFDB->add(_curKF);

    if (_enoughConsistentCandidates.empty()) {
        _curKF->setErase();
        return false;
    } else {
        return true;
    }

    _curKF->setErase();
    return false;
}

bool LoopClosing::computeSE3()
{
    const size_t nInitialCandidates = _enoughConsistentCandidates.size();

    Matcher matcher(0.75f, true);

    vector<Icp::Ptr> vpIcpSolvers;
    vpIcpSolvers.resize(nInitialCandidates);

    vector<vector<cv::DMatch>> vvDMatches;
    vvDMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates = 0; //candidates with enough matches

    for (size_t i = 0; i < nInitialCandidates; i++) {
        KeyFrame* pKF = _enoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->setNotErase();

        if (pKF->isBad()) {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.knnMatch(_curKF, pKF, vvDMatches[i]);

        // ORB-BRIEF 13
        // SURF-BRIEF 20
        // FAST-BRIEF 20
        if (nmatches < 20) {
            vbDiscarded[i] = true;
            continue;
        } else {
            Icp::Ptr icp(new Icp(pKF, _curKF, vvDMatches[i]));
            icp->setThreshold(0.08).setIterations(1000).refine(true);
            vpIcpSolvers[i] = icp;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform geometric validation
    for (size_t i = 0; i < nInitialCandidates; i++) {
        if (vbDiscarded[i])
            continue;

        KeyFrame* pKF = _enoughConsistentCandidates[i];

        Icp::Ptr icp = vpIcpSolvers[i];
        bool bOK = icp->compute();

        if (!bOK) {
            vbDiscarded[i] = true;
            nCandidates--;
        } else {
            vector<LandmarkPtr> vpLandmarkMatches(_curKF->_N, nullptr);
            for (const auto& m : icp->_inliers)
                vpLandmarkMatches[m.trainIdx] = pKF->getLandmark(m.queryIdx);

            Mat33 R = icp->_T.rotationMatrix();
            Vec3 t = icp->_T.translation();
            matcher.searchBySE3(_curKF, pKF, vpLandmarkMatches, R, t, 7.5);

            g2o::Sim3 gScm(R, t, 1.0);
            const int nInliers = BundleAdjustment::OptimizeSE3(_curKF, pKF, vpLandmarkMatches, gScm, 10.0);
            g2o::SE3Quat gTcm(gScm.rotation(), gScm.translation());

            // If optimization is succesful stop ransacs and continue
            if (nInliers >= 20) {
                bMatch = true;
                _matchedKF = pKF;
                g2o::SE3Quat gTmw(pKF->getPose().rotationMatrix(), pKF->getPose().translation());
                _g2oTcw = gTcm * gTmw;
                _Tcw = SE3(_g2oTcw.to_homogeneous_matrix());

                _curMatchedPoints = vpLandmarkMatches;
                break;
            }
        }
    }

    if (!bMatch) {
        for (size_t i = 0; i < nInitialCandidates; i++)
            _enoughConsistentCandidates[i]->setErase();
        _curKF->setErase();
        return false;
    }

    // Retrieve Landmarks seen in Loop Keyframe and neighbors
    vector<KeyFrame*> vpLoopConnectedKFs = _matchedKF->_node->getCovisibles();
    vpLoopConnectedKFs.push_back(_matchedKF);
    _loopLandmarks.clear();
    for (KeyFrame* pKF : vpLoopConnectedKFs) {
        vector<LandmarkPtr> vpLandmarks = pKF->getLandmarkMatches();
        for (LandmarkPtr pMP : vpLandmarks) {
            if (pMP) {
                if (!pMP->isBad() && pMP->_loopPointForKF != _curKF->_id) {
                    _loopLandmarks.push_back(pMP);
                    pMP->_loopPointForKF = _curKF->_id;
                }
            }
        }
    }

    // Find more matches projecting with the computed SE3
    matcher.searchByProjection(_curKF, _Tcw, _loopLandmarks, _curMatchedPoints, 10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for (LandmarkPtr pMP : _curMatchedPoints) {
        if (pMP)
            nTotalMatches++;
    }

    if (nTotalMatches >= 40) {
        for (size_t i = 0; i < nInitialCandidates; i++)
            if (_enoughConsistentCandidates[i] != _matchedKF)
                _enoughConsistentCandidates[i]->setErase();
        return true;
    } else {
        for (size_t i = 0; i < nInitialCandidates; i++)
            _enoughConsistentCandidates[i]->setErase();
        _curKF->setErase();
        return false;
    }
}

void LoopClosing::correctLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    _localMapper->requestStop();

    // If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA()) {
        unique_lock<mutex> lock(_mutexGBA);
        _stopGBA = true;

        _fullBAIdx++;

        if (_threadGBA) {
            _threadGBA->detach();
            delete _threadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while (!_localMapper->isStopped()) {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    _curKF->_node->updateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected SE3 pose by propagation
    _curConnectedKFs = _curKF->_node->getCovisibles();
    _curConnectedKFs.push_back(_curKF);

    KeyFrameAndPose CorrectedSE3, NonCorrectedSE3;
    CorrectedSE3[_curKF] = _g2oTcw;
    SE3 Twc = _curKF->getPoseInverse();

    {
        // Get Map Mutex
        unique_lock<mutex> lock(_map->_mutexMapUpdate);

        for (KeyFrame* pKFi : _curConnectedKFs) {
            SE3 Tiw = pKFi->getPose();

            if (pKFi != _curKF) {
                SE3 Tic = Tiw * Twc;
                Mat33 Ric = Tic.rotationMatrix();
                Vec3 tic = Tic.translation();
                g2o::SE3Quat g2oTic(Ric, tic);
                g2o::SE3Quat g2oCorrectedTiw = g2oTic * _g2oTcw;
                // Pose corrected with the SE3 of the loop closure
                CorrectedSE3[pKFi] = g2oCorrectedTiw;
            }

            Mat33 Riw = Tiw.rotationMatrix();
            Vec3 tiw = Tiw.translation();
            g2o::SE3Quat g2oTiw(Riw, tiw);
            // Pose without correction
            NonCorrectedSE3[pKFi] = g2oTiw;
        }

        // Correct all Landmarks obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for (auto& [pKFi, g2oCorrectedTiw] : CorrectedSE3) {
            g2o::SE3Quat g2oCorrectedTwi = g2oCorrectedTiw.inverse();
            g2o::SE3Quat g2oTiw = NonCorrectedSE3[pKFi];

            vector<LandmarkPtr> vpMPsi = pKFi->getLandmarkMatches();
            for (LandmarkPtr pMPi : vpMPsi) {
                if (!pMPi)
                    continue;
                if (pMPi->isBad())
                    continue;
                if (pMPi->_correctedByKF == _curKF->_id)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                Vec3 P3Dw = pMPi->getWorldPos();
                Vec3 eigCorrectedP3Dw = g2oCorrectedTwi.map(g2oTiw.map(P3Dw));

                pMPi->setWorldPos(eigCorrectedP3Dw);
                pMPi->_correctedByKF = _curKF->_id;
                pMPi->_correctedReference = pKFi->_id;
                pMPi->updateNormalAndDepth();
            }

            // Update keyframe pose with corrected SE3
            Mat33 eigR = g2oCorrectedTiw.rotation().toRotationMatrix();
            Vec3 eigt = g2oCorrectedTiw.translation();

            pKFi->setPose(SE3(eigR, eigt));

            // Make sure connections are updated
            pKFi->_node->updateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for (size_t i = 0; i < _curMatchedPoints.size(); i++) {
            if (_curMatchedPoints[i]) {
                LandmarkPtr pLoopMP = _curMatchedPoints[i];
                LandmarkPtr pCurMP = _curKF->getLandmark(i);
                if (pCurMP)
                    pCurMP->replace(pLoopMP);
                else {
                    _curKF->addLandmark(pLoopMP, i);
                    pLoopMP->addObservation(_curKF, i);
                    pLoopMP->computeDistinctiveDescriptors();
                }
            }
        }
    }

    // Project Landmarks observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    searchAndFuse(CorrectedSE3);

    // After the Landmark fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*>> LoopConnections;

    for (KeyFrame* pKFi : _curConnectedKFs) {
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->_node->getCovisibles();

        // Update connections. Detect new links.
        pKFi->_node->updateConnections();
        LoopConnections[pKFi] = pKFi->_node->getConnectedKFs();
        for (KeyFrame* pPrevKF : vpPreviousNeighbors)
            LoopConnections[pKFi].erase(pPrevKF);

        for (KeyFrame* pKF2 : _curConnectedKFs)
            LoopConnections[pKFi].erase(pKF2);
    }

    // Optimize graph
    PoseGraph::optimize(_map, _matchedKF, _curKF, NonCorrectedSE3, CorrectedSE3, LoopConnections);

    _map->informNewBigChange();

    // Add loop edge
    _matchedKF->_node->addLoopEdge(_curKF);
    _curKF->_node->addLoopEdge(_matchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    _runningGBA = true;
    _finishedGBA = false;
    _stopGBA = false;
    _threadGBA = new thread(&LoopClosing::runGlobalBundleAdjustment, this, _curKF->_id);

    // Loop closed. Release Local Mapping.
    _localMapper->release();

    _prevLoopKFid = _curKF->_id;
}

void LoopClosing::searchAndFuse(const KeyFrameAndPose& CorrectedPosesMap)
{
    Matcher matcher(0.8f);

    for (auto& [pKF, g2oTcw] : CorrectedPosesMap) {
        SE3 cvTcw(g2oTcw.to_homogeneous_matrix());

        vector<LandmarkPtr> vpReplacePoints(_loopLandmarks.size(), nullptr);
        matcher.fuse(pKF, cvTcw, _loopLandmarks, 4, vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(_map->_mutexMapUpdate);
        const size_t nLP = _loopLandmarks.size();
        for (size_t i = 0; i < nLP; i++) {
            LandmarkPtr pRep = vpReplacePoints[i];
            if (pRep) {
                pRep->replace(_loopLandmarks[i]);
            }
        }
    }
}

void LoopClosing::requestReset()
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
        usleep(5000);
    }
}

void LoopClosing::resetIfRequested()
{
    unique_lock<mutex> lock(_mutexReset);
    if (_resetRequested) {
        _KFsQueue.clear();
        _prevLoopKFid = 0;
        _resetRequested = false;
    }
}

void LoopClosing::runGlobalBundleAdjustment(int nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx = _fullBAIdx;
    BundleAdjustment::Global(_map, 10, &_stopGBA, nLoopKF, false);

    // Update all Landmarks and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(_mutexGBA);
        if (idx != _fullBAIdx)
            return;

        if (!_stopGBA) {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            _localMapper->requestStop();
            // Wait until Local Mapping has effectively stopped

            while (!_localMapper->isStopped() && !_localMapper->isFinished()) {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(_map->_mutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(_map->_keyFrameOrigins.begin(), _map->_keyFrameOrigins.end());

            while (!lpKFtoCheck.empty()) {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->_node->getChildrens();
                SE3 Twc = pKF->getPoseInverse();
                for (KeyFrame* pChild : sChilds) {
                    if (pChild->_BAGlobalForKF != nLoopKF) {
                        SE3 Tchildc = pChild->getPose() * Twc;
                        pChild->_TcwGBA = Tchildc * pKF->_TcwGBA;
                        pChild->_BAGlobalForKF = nLoopKF;
                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->_TcwBefGBA = pKF->getPose();
                pKF->setPose(pKF->_TcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct Landmarks
            const vector<LandmarkPtr> vpMPs = _map->getAllLandmarks();

            for (LandmarkPtr pMP : vpMPs) {
                if (pMP->isBad())
                    continue;

                if (pMP->_BAGlobalForKF == nLoopKF) {
                    // If optimized by Global BA, just update
                    pMP->setWorldPos(pMP->_posGBA);
                } else {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->getReferenceKeyFrame();

                    if (pRefKF->_BAGlobalForKF != nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    Vec3 Xc = pRefKF->_TcwBefGBA * pMP->getWorldPos();

                    // Backproject using corrected camera
                    SE3 Twc = pRefKF->getPoseInverse();
                    pMP->setWorldPos(Twc * Xc);
                }
            }

            _map->informNewBigChange();

            _localMapper->release();

            cout << "Map updated!" << endl;
        }

        _finishedGBA = true;
        _runningGBA = false;
    }
}

void LoopClosing::requestFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finishRequested = true;
}

bool LoopClosing::checkFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finishRequested;
}

void LoopClosing::setFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finished;
}

void LoopClosing::join()
{
    if (_thread.joinable()) {
        _thread.join();
        cout << "Loop Closing thread JOINED" << endl;
    }
}
