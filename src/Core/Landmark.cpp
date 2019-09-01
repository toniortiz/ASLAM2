#include "Landmark.h"
#include "Feature.h"
#include "Features/Matcher.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include <mutex>

using namespace std;

int Landmark::_nextId = 0;
mutex Landmark::_globalMutex;

Landmark::Landmark(const Vec3& Pos, KeyFrame* pRefKF, MapPtr pMap)
    : _firstKFid(pRefKF->_id)
    , _firstFrame(pRefKF->_frameId)
    , _nobs(0)
    , _trackReferenceForFrame(0)
    , _lastFrameSeen(0)
    , _BALocalForKF(0)
    , _fuseCandidateForKF(0)
    , _loopPointForKF(0)
    , _correctedByKF(0)
    , _correctedReference(0)
    , _BAGlobalForKF(0)
    , _refKF(pRefKF)
    , _visible(1)
    , _found(1)
    , _bad(false)
    , _replaced(nullptr)
    , _minDistance(0)
    , _maxDistance(0)
    , _map(pMap)
{
    _worldPos = Pos;
    _normal = Vec3::Zero();

    _id = _nextId++;
}

Landmark::Landmark(const Vec3& Pos, MapPtr pMap, Frame* pFrame, const int& idxF)
    : _firstKFid(-1)
    , _firstFrame(pFrame->_id)
    , _nobs(0)
    , _trackReferenceForFrame(0)
    , _lastFrameSeen(0)
    , _BALocalForKF(0)
    , _fuseCandidateForKF(0)
    , _loopPointForKF(0)
    , _correctedByKF(0)
    , _correctedReference(0)
    , _BAGlobalForKF(0)
    , _refKF(nullptr)
    , _visible(1)
    , _found(1)
    , _bad(false)
    , _replaced(nullptr)
    , _map(pMap)
{
    _worldPos = Pos;
    Vec3 Ow = pFrame->getPoseInverse().translation();
    _normal = _worldPos - Ow;
    _normal = _normal / _normal.norm();

    Vec3 PC = Pos - Ow;
    const double dist = PC.norm();
    const int level = pFrame->_features[idxF]->_level;
    const double levelScaleFactor = pFrame->_scaleFactors[level];
    const int nLevels = pFrame->_scaleLevels;

    _maxDistance = dist * levelScaleFactor;
    _minDistance = _maxDistance / pFrame->_scaleFactors[nLevels - 1];

    pFrame->_descriptors.row(idxF).copyTo(_descriptor);

    _id = _nextId++;
}

void Landmark::setWorldPos(const Vec3& Pos)
{
    unique_lock<mutex> lock2(_globalMutex);
    unique_lock<mutex> lock(_mutexPos);
    _worldPos = Pos;
}

Vec3 Landmark::getWorldPos()
{
    unique_lock<mutex> lock(_mutexPos);
    return _worldPos;
}

Vec3 Landmark::getNormal()
{
    unique_lock<mutex> lock(_mutexPos);
    return _normal;
}

KeyFrame* Landmark::getReferenceKeyFrame()
{
    unique_lock<mutex> lock(_mutexFeatures);
    return _refKF;
}

void Landmark::addObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(_mutexFeatures);
    if (_observations.count(pKF))
        return;
    _observations[pKF] = idx;

    if (pKF->_features[idx]->_right >= 0)
        _nobs += 2;
    else
        _nobs++;
}

void Landmark::eraseObservation(KeyFrame* pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(_mutexFeatures);
        if (_observations.count(pKF)) {
            size_t idx = _observations[pKF];
            if (pKF->_features[idx]->_right >= 0)
                _nobs -= 2;
            else
                _nobs--;

            _observations.erase(pKF);

            if (_refKF == pKF)
                _refKF = _observations.begin()->first;

            // If only 2 observations or less, discard point
            if (_nobs <= 2)
                bBad = true;
        }
    }

    if (bBad)
        setBadFlag();
}

map<KeyFrame*, size_t> Landmark::getObservations()
{
    unique_lock<mutex> lock(_mutexFeatures);
    return _observations;
}

int Landmark::observations()
{
    unique_lock<mutex> lock(_mutexFeatures);
    return _nobs;
}

void Landmark::setBadFlag()
{
    map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(_mutexFeatures);
        unique_lock<mutex> lock2(_mutexPos);
        _bad = true;
        obs = _observations;
        _observations.clear();
    }
    for (auto& [pKF, idx] : obs)
        pKF->eraseLandmarkMatch(idx);

    _map->eraseLandmark(this->_id);
}

LandmarkPtr Landmark::getReplaced()
{
    unique_lock<mutex> lock1(_mutexFeatures);
    unique_lock<mutex> lock2(_mutexPos);
    return _replaced;
}

void Landmark::replace(LandmarkPtr pMP)
{
    if (pMP->_id == this->_id)
        return;

    int nvisible, nfound;
    map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(_mutexFeatures);
        unique_lock<mutex> lock2(_mutexPos);
        obs = _observations;
        _observations.clear();
        _bad = true;
        nvisible = _visible;
        nfound = _found;
        _replaced = pMP;
    }

    for (auto& [pKF, idx] : obs) {
        // Replace measurement in keyframe
        if (!pMP->isInKeyFrame(pKF)) {
            pKF->replaceLandmarkMatch(idx, pMP);
            pMP->addObservation(pKF, idx);
        } else {
            pKF->eraseLandmarkMatch(idx);
        }
    }
    pMP->increaseFound(nfound);
    pMP->increaseVisible(nvisible);
    pMP->computeDistinctiveDescriptors();

    _map->eraseLandmark(this->_id);
}

bool Landmark::isBad()
{
    unique_lock<mutex> lock(_mutexFeatures);
    unique_lock<mutex> lock2(_mutexPos);
    return _bad;
}

void Landmark::increaseVisible(int n)
{
    unique_lock<mutex> lock(_mutexFeatures);
    _visible += n;
}

void Landmark::increaseFound(int n)
{
    unique_lock<mutex> lock(_mutexFeatures);
    _found += n;
}

double Landmark::getFoundRatio()
{
    unique_lock<mutex> lock(_mutexFeatures);
    return static_cast<double>(_found) / _visible;
}

void Landmark::computeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;
    map<KeyFrame*, size_t> observations;

    {
        unique_lock<mutex> lock1(_mutexFeatures);
        if (_bad)
            return;
        observations = _observations;
    }

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for (auto& [pKF, idx] : observations) {
        if (!pKF->isBad())
            vDescriptors.push_back(pKF->_descriptors.row(static_cast<int>(idx)));
    }

    if (vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for (size_t i = 0; i < N; i++) {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++) {
            int distij = Matcher::descriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = numeric_limits<int>::max();
    int BestIdx = 0;
    for (size_t i = 0; i < N; i++) {
        vector<int> vDists(Distances[i], Distances[i] + N);
        sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];

        if (median < BestMedian) {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(_mutexFeatures);
        _descriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat Landmark::getDescriptor()
{
    unique_lock<mutex> lock(_mutexFeatures);
    return _descriptor.clone();
}

int Landmark::getIndexInKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(_mutexFeatures);
    if (_observations.count(pKF))
        return _observations[pKF];
    else
        return -1;
}

bool Landmark::isInKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(_mutexFeatures);
    return (_observations.count(pKF));
}

void Landmark::updateNormalAndDepth()
{
    map<KeyFrame*, size_t> observations;
    KeyFrame* pRefKF;
    Vec3 Pos;
    {
        unique_lock<mutex> lock1(_mutexFeatures);
        unique_lock<mutex> lock2(_mutexPos);
        if (_bad)
            return;
        observations = _observations;
        pRefKF = _refKF;
        Pos = _worldPos;
    }

    if (observations.empty())
        return;

    Vec3 normal = Vec3::Zero();
    int n = 0;
    for (auto& [pKF, idx] : observations) {
        Vec3 Owi = pKF->getPoseInverse().translation();
        Vec3 normali = _worldPos - Owi;
        normal = normal + normali / normali.norm();
        n++;
    }

    Vec3 PC = Pos - pRefKF->getPoseInverse().translation();
    const double dist = PC.norm();
    const int level = pRefKF->_features[observations[pRefKF]]->_level;
    const double levelScaleFactor = pRefKF->_scaleFactors[level];
    const int nLevels = pRefKF->_scaleLevels;

    {
        unique_lock<mutex> lock3(_mutexPos);
        _maxDistance = dist * levelScaleFactor;
        _minDistance = _maxDistance / pRefKF->_scaleFactors[nLevels - 1];
        _normal = normal / n;
    }
}

double Landmark::getMinDistanceInvariance()
{
    unique_lock<mutex> lock(_mutexPos);
    return 0.8 * _minDistance;
}

double Landmark::getMaxDistanceInvariance()
{
    unique_lock<mutex> lock(_mutexPos);
    return 1.2 * _maxDistance;
}

int Landmark::predictScale(const double& currentDist, KeyFrame* pKF)
{
    double ratio;
    {
        unique_lock<mutex> lock(_mutexPos);
        ratio = _maxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pKF->_logScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pKF->_scaleLevels)
        nScale = pKF->_scaleLevels - 1;

    return nScale;
}

int Landmark::predictScale(const double& currentDist, Frame* pF)
{
    double ratio;
    {
        unique_lock<mutex> lock(_mutexPos);
        ratio = _maxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pF->_logScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pF->_scaleLevels)
        nScale = pF->_scaleLevels - 1;

    return nScale;
}
