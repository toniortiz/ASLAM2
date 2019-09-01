#include "Tracking.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/Map.h"
#include "Core/PinholeCamera.h"
#include "Drawer/MapDrawer.h"
#include "Drawer/Viewer.h"
#include "Features/Extractor.h"
#include "Features/Matcher.h"
#include "IO/Dataset.h"
#include "Mapping.h"
#include "PlaceRecognition/KeyFrameDatabase.h"
#include "PlaceRecognition/LoopClosing.h"
#include "Solver/PnPRansac.h"
#include "Solver/SE3solver.h"
#include "System.h"
#include "System/LocalMap.h"
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <unistd.h>

using namespace std;

Tracking::Tracking(System* pSys, ExtractorPtr pExtractor, VocabularyPtr pVoc, MapDrawerPtr pMapDrawer,
    MapPtr pMap, KeyFrameDatabasePtr pKFDB, DatasetPtr pDataset)
    : _state(NO_IMAGES_YET)
    , _extractor(pExtractor)
    , _vocabulary(pVoc)
    , _KFDB(pKFDB)
    , _system(pSys)
    , _viewer(nullptr)
    , _mapDrawer(pMapDrawer)
    , _map(pMap)
{
    _camera = pDataset->camera();
    _speed = SE3(Mat44::Identity());

    _localMap.reset(new LocalMap(_camera));
}

void Tracking::setLocalMapper(MappingPtr pLocalMapper) { _localMapper = pLocalMapper; }

void Tracking::setLoopClosing(LoopClosingPtr pLoopClosing) { _loopClosing = pLoopClosing; }

void Tracking::setViewer(ViewerPtr pViewer) { _viewer = pViewer; }

SE3 Tracking::grabImageRGBD(const cv::Mat& imBGR, const cv::Mat& imD, const double& timestamp)
{
    _curFrame = make_shared<Frame>(imBGR, imD, timestamp, _extractor, _vocabulary, _camera);

    track();

    return _curFrame->getPose();
}

void Tracking::track()
{
    if (_state == NO_IMAGES_YET)
        _state = NOT_INITIALIZED;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(_map->_mutexMapUpdate);

    if (_state == NOT_INITIALIZED) {
        initialization();

        if (_state != OK)
            return;
    } else {
        bool bOK;

        if (_state == OK) {
            checkReplacedInLastFrame();

            if (_speed.matrix().isIdentity())
                bOK = trackReferenceKeyFrame();
            else
                bOK = trackWithMotionModel();
        } else
            bOK = false;

        _curFrame->_refKF = _localMap->_refKF;

        if (bOK) {
            // This is for visualization
            _map->setReferenceLandmarks(_localMap->_localPoints);

            bOK = _localMap->track(_curFrame);

            unique_lock<mutex> locks(_stats.mMutex);
            _stats.nLocalKFs = _localMap->_localKFs.size();
            _stats.nLocalMPs = _localMap->_localPoints.size();
        }

        if (bOK)
            _state = OK;
        else
            _state = LOST;

        // If tracking were good, check if we insert a keyframe
        if (bOK) {
            updateMotionModel();

            _mapDrawer->setCurrentCameraPose(_curFrame->getPose());

            cleanOdometryMatches();

            // Check if we need to insert a new keyframe
            if (needNewKeyFrame())
                createNewKeyFrame();

            for (size_t i = 0; i < _curFrame->_N; i++) {
                if (_curFrame->_features[i]->_point && _curFrame->_features[i]->isOutlier())
                    _curFrame->eraseLandmark(i);
            }
        }

        // Reset if the camera get lost soon after initialization
        if (_state == LOST) {
            if (_map->KeyFramesInMap() <= 5) {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                _system->reset();
                return;
            }
        }

        if (!_curFrame->_refKF)
            _curFrame->_refKF = _localMap->_refKF;

        _prevFrame = _curFrame;
    }

    updateRelativePoses();
}

void Tracking::updateMotionModel()
{
    if (_state != LOST)
        _speed = _curFrame->getPose() * _prevFrame->getPoseInverse();
    else
        _speed = SE3(Mat44::Identity());
}

void Tracking::cleanOdometryMatches()
{
    // Clean VO matches
    for (size_t i = 0; i < _curFrame->_N; i++) {
        if (_curFrame->_features[i]->_point)
            if (_curFrame->_features[i]->_point->observations() < 1) {
                _curFrame->_features[i]->setInlier();
                _curFrame->eraseLandmark(i);
            }
    }
}

void Tracking::updateRelativePoses()
{
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if (_state != LOST) {
        SE3 Tcr = _curFrame->getPose() * _curFrame->_refKF->getPoseInverse();
        _relativeFramePoses.push_back(Tcr);
        _references.push_back(_localMap->_refKF);
        _frameTimes.push_back(_curFrame->_timestamp);
        _losts.push_back(_state == LOST);
    } else {
        // This can happen if tracking is lost
        _relativeFramePoses.push_back(_relativeFramePoses.back());
        _references.push_back(_references.back());
        _frameTimes.push_back(_frameTimes.back());
        _losts.push_back(_state == LOST);
    }
}

void Tracking::initialization()
{
    // Set Frame pose to the origin
    _curFrame->setPose(SE3(Mat44::Identity()));

    KeyFrame* pKFini = new KeyFrame(*_curFrame, _map, _KFDB);

    _map->addKeyFrame(pKFini);

    // Create Landmarks and asscoiate to KeyFrame
    for (size_t i = 0; i < _curFrame->_N; i++) {
        if (_curFrame->_features[i]->isValid()) {
            Vec3 xc = _curFrame->_features[i]->_Xc;
            Vec3 xw = _curFrame->getPoseInverse() * xc;
            LandmarkPtr pNewMP(new Landmark(xw, pKFini, _map));
            pNewMP->addObservation(pKFini, i);
            pKFini->addLandmark(pNewMP, i);
            pNewMP->computeDistinctiveDescriptors();
            pNewMP->updateNormalAndDepth();
            _map->addLandmark(pNewMP);

            _curFrame->addLandmark(i, pNewMP);
        }
    }

    cout << "New map created with " << _map->LandmarksInMap() << " points" << endl;

    _localMapper->insertKeyFrame(pKFini);

    _prevFrame = _curFrame;
    _prevKFid = _curFrame->_id;
    _prevKF = pKFini;

    _localMap->_refKF = pKFini;
    _localMap->_localPoints = _map->getAllLandmarks();
    _curFrame->_refKF = pKFini;

    _map->setReferenceLandmarks(_localMap->_localPoints);
    _map->_keyFrameOrigins.push_back(pKFini);

    _mapDrawer->setCurrentCameraPose(_curFrame->getPose());

    _state = OK;

    unique_lock<mutex> locks(_stats.mMutex);
    _stats.acumObservations += _curFrame->_N;
}

void Tracking::checkReplacedInLastFrame()
{
    for (size_t i = 0; i < _prevFrame->_N; i++) {
        LandmarkPtr pMP = _prevFrame->_features[i]->_point;

        if (pMP) {
            LandmarkPtr pRep = pMP->getReplaced();
            if (pRep)
                _prevFrame->addLandmark(i, pRep);
        }
    }
}

bool Tracking::trackReferenceKeyFrame()
{
    Matcher matcher(0.7f, true);
    vector<LandmarkPtr> vpLandmarkMatches;
    std::vector<cv::DMatch> vMatches;

    int nmatches = matcher.knnMatch(_localMap->_refKF, _curFrame, vpLandmarkMatches, vMatches);

    if (nmatches < 15)
        return false;

    for (size_t i = 0; i < vpLandmarkMatches.size(); ++i)
        _curFrame->_features[i]->_point = vpLandmarkMatches[i];

    _curFrame->setPose(_prevFrame->getPose());

    PnPRansac pnp(_curFrame);
    pnp.compute();

    {
        unique_lock<mutex> locks(_stats.mMutex);
        _stats.acumMatches += vMatches.size();
        _stats.acumObservations += _curFrame->_N;
        _stats.acumInliers += nmatches;
        _stats.acumInlierRatio += (double(nmatches) / double(vMatches.size()));
    }

    return true;
}

void Tracking::updatePrevFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = _prevFrame->_refKF;
    SE3 Tlr = _relativeFramePoses.back();

    _prevFrame->setPose(Tlr * pRef->getPose());

    if (_prevKFid == _prevFrame->_id)
        return;

    createNewLandmarks(_prevFrame);
}

bool Tracking::trackWithMotionModel()
{
    updatePrevFrame();

    _curFrame->setPose(_speed * _prevFrame->getPose());
    _curFrame->eraseLandmarks();

    // Project points seen in previous frame
    Matcher matcher(0.9f, true);
    vector<cv::DMatch> matches;
    int nmatches = matcher.searchByProjection(_prevFrame, _curFrame, matches, 15);

    // If motion model is violated, uses a bruteforce search
    if (nmatches < 20) {
        _curFrame->eraseLandmarks();
        matches.clear();
        nmatches = matcher.knnMatch(_prevFrame, _curFrame, matches);
    }

    if (nmatches < 20)
        return false;

    SE3solver ransac(_prevFrame, _curFrame, matches);
    ransac.setRansacParameters(200, 20, 3.0f, 4);
    bool bOK = ransac.compute();

    _viewer->setRansacError(ransac._rmse);

    /*
    double tau_1 = 5;
    double tau_2 = 20;
    double miu = 20;
    double acumres = 0.0;
    double meanres = 0.0;
    for (auto& m : ransac._inliers) {
        const cv::Point3f& src = _prevFrame->getCameraPoint(m.queryIdx);
        const cv::Point3f& tgt = _curFrame->getCameraPoint(m.trainIdx);

        cv::Mat xs = (cv::Mat_<float>(3, 1) << src.x, src.y, src.z);
        cv::Mat xt = (cv::Mat_<float>(3, 1) << tgt.x, tgt.y, tgt.z);

        cv::Mat T = ransac._T;
        cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
        cv::Mat t = T.rowRange(0, 3).col(3);

        cv::Mat xs_t = R * xs + t;
        double res = cv::norm(xs_t, xt, cv::NORM_L2SQR);
        acumres += res;
    }
    meanres = acumres / ransac._inliers.size();
    cout << meanres << endl;
*/
    /*
        if (bOK) {
            if (ransac._rmse > 1.0f) {
                if (ransac._rmse > 1.5f) {
                    Gicp gicp(_prevFrame.get(), _curFrame.get(), matches);
                    gicp.setIterations(15).setCorrespondenceDistance(0.08);
                    cv::Mat guess = cv::Mat::eye(4, 4, CV_32F);
                    bOK = gicp.compute(guess);
                    _viewer->setIcpScore(gicp._score);
                } else {
                    Gicp gicp(_prevFrame.get(), _curFrame.get(), ransac._inliers);
                    gicp.setIterations(8).setCorrespondenceDistance(0.07);
                    bOK = gicp.compute(ransac._T);
                    _viewer->setIcpScore(gicp._score);
                }
            }
        } else {
            Gicp gicp(_prevFrame.get(), _curFrame.get(), matches);
            gicp.setIterations(20).setCorrespondenceDistance(0.08);
            cv::Mat guess = cv::Mat::eye(4, 4, CV_32F);
            bOK = gicp.compute(guess);
            _viewer->setIcpScore(gicp._score);
        }
*/

    {
        unique_lock<mutex> locks(_stats.mMutex);
        _stats.acumMatches += matches.size();
        _stats.meanMatches = _stats.acumMatches / _curFrame->_id;
        _stats.acumObservations += _curFrame->_N;
        _stats.meanObservations = _stats.acumObservations / _curFrame->_id;
        _stats.acumInliers += ransac._inliers.size();
        _stats.meanInliers = _stats.acumInliers / _curFrame->_id;
        _stats.acumInlierRatio += (double(ransac._inliers.size()) / double(matches.size()));
        _stats.meanInlierRatio = _stats.acumInlierRatio / _curFrame->_id;
    }

    {
        unique_lock<mutex> lock(_mutexImages);
        _imgMatches = Matcher::getImageMatches(_prevFrame, _curFrame, ransac._inliers);
    }

    return bOK;
}

void Tracking::createNewLandmarks(FramePtr frame, KeyFrame* pKF)
{
    for (size_t i = 0; i < frame->_N; i++) {
        bool bCreateNew = false;

        if (frame->_features[i]->isValid()) {
            if (!frame->_features[i]->_point) {
                bCreateNew = true;
            } else if (frame->_features[i]->_point->observations() < 1) {
                bCreateNew = true;
                if (pKF)
                    frame->eraseLandmark(i);
            }
        }

        if (bCreateNew) {
            LandmarkPtr pNewMP = nullptr;
            Vec3 xc = frame->_features[i]->_Xc;
            Vec3 xw = frame->getPoseInverse() * xc;

            if (pKF) {
                pNewMP.reset(new Landmark(xw, pKF, _map));
                pNewMP->addObservation(pKF, i);
                pKF->addLandmark(pNewMP, i);
                pNewMP->computeDistinctiveDescriptors();
                pNewMP->updateNormalAndDepth();
                _map->addLandmark(pNewMP);
            } else {
                pNewMP.reset(new Landmark(xw, _map, frame.get(), i));
            }

            frame->addLandmark(i, pNewMP);
        }
    }
}

double tnorm(const SE3& T)
{
    Vec3 t = T.translation();
    return t.norm();
}

double rnorm(const SE3& T)
{
    Mat33 R = T.rotationMatrix();
    return acos(0.5 * (R(0, 0) + R(1, 1) + R(2, 2) - 1.0));
}

bool Tracking::needNewKeyFrame()
{
    // New keyframes are added when the accumulated motion since the previous
    // keyframe exceeds either 10° in rotation or 20 cm in translation
    static const double mint = 0.20; // m
    static const double minr = 0.1745; // rad

    if (_localMapper->isStopped() || _localMapper->stopRequested())
        return false;

    bool bLocalMappingIdle = _localMapper->acceptKeyFrames();

    SE3 delta = _curFrame->getPoseInverse() * _prevKF->getPose();
    bool c1 = tnorm(delta) > mint;
    bool c2 = rnorm(delta) > minr;

    if (c1 || c2) {
        if (bLocalMappingIdle) {
            return true;
        } else {
            _localMapper->interruptBA();
            if (_localMapper->KeyframesInQueue() < 3)
                return true;
            else
                return false;
        }
    } else
        return false;
}

void Tracking::createNewKeyFrame()
{
    if (!_localMapper->setNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(*_curFrame, _map, _KFDB);

    _localMap->_refKF = pKF;
    _curFrame->_refKF = pKF;

    createNewLandmarks(_curFrame, pKF);

    _localMapper->insertKeyFrame(pKF);
    _localMapper->setNotStop(false);

    _prevKFid = _curFrame->_id;
    _prevKF = pKF;
}

int Tracking::meanInliers()
{
    unique_lock<mutex> lock(_stats.mMutex);
    return _stats.meanInliers;
}

double Tracking::meanInlierRatio()
{
    unique_lock<mutex> lock(_stats.mMutex);
    return _stats.meanInlierRatio;
}

int Tracking::meanObservations()
{
    unique_lock<mutex> lock(_stats.mMutex);
    return _stats.meanObservations;
}

int Tracking::meanMatches()
{
    unique_lock<mutex> lock(_stats.mMutex);
    return _stats.meanMatches;
}

int Tracking::localKFs()
{
    unique_lock<mutex> lock(_stats.mMutex);
    return _stats.nLocalKFs;
}

cv::Mat Tracking::getImageMatches()
{
    unique_lock<mutex> lock(_mutexImages);
    return _imgMatches.clone();
}

int Tracking::localMPs()
{
    unique_lock<mutex> lock(_stats.mMutex);
    return _stats.nLocalMPs;
}

void Tracking::reset()
{
    cout << "System Reseting" << endl;
    if (_viewer) {
        _viewer->requestStop();
        while (!_viewer->isStopped())
            usleep(3000);
    }

    cout << "Reseting Local Mapper...";
    _localMapper->requestReset();
    cout << " done" << endl;

    cout << "Reseting Loop Closing...";
    _loopClosing->requestReset();
    cout << " done" << endl;

    cout << "Reseting Database...";
    _KFDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase Landmarks and KeyFrames)
    _map->clear();

    KeyFrame::_nextId = 0;
    Frame::_nextId = 0;
    _state = NO_IMAGES_YET;

    _relativeFramePoses.clear();
    _references.clear();
    _frameTimes.clear();
    _losts.clear();

    if (_viewer)
        _viewer->release();
}
