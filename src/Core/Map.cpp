#include "Map.h"
#include "KeyFrame.h"
#include "Landmark.h"
#include <mutex>

using namespace std;

Map::Map()
    : _maxKFid(0)
    , _bigChangeIdx(0)
{
}

void Map::addKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(_mutexMap);
    //    _keyFrames.insert(pKF);
    _keyframes.insert({ pKF->_id, pKF });
    if (pKF->_id > _maxKFid)
        _maxKFid = pKF->_id;
}

void Map::addLandmark(LandmarkPtr pMP)
{
    unique_lock<mutex> lock(_mutexMap);
    //    _landmarks.insert(pMP);
    _landmarks.insert({ pMP->_id, pMP });
}

void Map::eraseLandmark(LandmarkPtr pMP)
{
    unique_lock<mutex> lock(_mutexMap);
    _landmarks.erase(pMP->_id);
}

void Map::eraseLandmark(const int& id)
{
    unique_lock<mutex> lock(_mutexMap);
    _landmarks.erase(id);
}

void Map::eraseKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(_mutexMap);
    _keyframes.erase(pKF->_id);
}

void Map::eraseKeyFrame(const int& id)
{
    unique_lock<mutex> lock(_mutexMap);
    _keyframes.erase(id);
}

void Map::setReferenceLandmarks(const vector<LandmarkPtr>& vpMPs)
{
    unique_lock<mutex> lock(_mutexMap);
    _referenceLandmarks = vpMPs;
}

void Map::informNewBigChange()
{
    unique_lock<mutex> lock(_mutexMap);
    _bigChangeIdx++;
}

int Map::getLastBigChangeIdx()
{
    unique_lock<mutex> lock(_mutexMap);
    return _bigChangeIdx;
}

vector<KeyFrame*> Map::getAllKeyFrames()
{
    unique_lock<mutex> lock(_mutexMap);
    vector<KeyFrame*> kfs;
    kfs.reserve(_keyframes.size());
    for (auto& [id, pKF] : _keyframes)
        kfs.push_back(pKF);
    return kfs;
}

vector<LandmarkPtr> Map::getAllLandmarks()
{
    unique_lock<mutex> lock(_mutexMap);
    vector<LandmarkPtr> lmks;
    lmks.reserve(_landmarks.size());
    for (auto& [id, pMP] : _landmarks)
        lmks.push_back(pMP);
    return lmks;
}

size_t Map::LandmarksInMap()
{
    unique_lock<mutex> lock(_mutexMap);
    return _landmarks.size();
}

size_t Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(_mutexMap);
    return _keyframes.size();
}

vector<LandmarkPtr> Map::getReferenceLandmarks()
{
    unique_lock<mutex> lock(_mutexMap);
    return _referenceLandmarks;
}

int Map::getMaxKFid()
{
    unique_lock<mutex> lock(_mutexMap);
    return _maxKFid;
}

void Map::clear()
{
    for (auto& [id, pKF] : _keyframes)
        delete pKF;

    _landmarks.clear();
    _keyframes.clear();
    _maxKFid = 0;
    _referenceLandmarks.clear();
    _keyFrameOrigins.clear();
}
