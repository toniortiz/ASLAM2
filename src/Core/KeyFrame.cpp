#include "KeyFrame.h"
#include "Feature.h"
#include "Frame.h"
#include "GraphNode.h"
#include "Landmark.h"
#include "Map.h"
#include "PinholeCamera.h"
#include "PlaceRecognition/KeyFrameDatabase.h"
#include "System/Converter.h"
#include <boost/make_shared.hpp>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

int KeyFrame::_nextId = 0;

KeyFrame::KeyFrame(Frame& F, MapPtr pMap, KeyFrameDatabasePtr pKFDB)
    : _colorIm(F._colorIm.clone())
    , _grayIm(F._grayIm.clone())
    , _depthIm(F._depthIm.clone())
    , _pointCloud(nullptr)
    , _frameId(F._id)
    , _timestamp(F._timestamp)
    , _gridCols(FRAME_GRID_COLS)
    , _gridRows(FRAME_GRID_ROWS)
    , _gridElementWidthInv(F._gridElementWidthInv)
    , _gridElementHeightInv(F._gridElementHeightInv)
    , _trackReferenceForFrame(0)
    , _fuseTargetForKF(0)
    , _BALocalForKF(0)
    , _BAFixedForKF(0)
    , _loopQuery(0)
    , _loopWords(0)
    , _BAGlobalForKF(0)
    , _N(F._N)
    , _keys(F._keys)
    , _descriptors(F._descriptors.clone())
    , _features(F._features)
    , _bowVec(F._bowVec)
    , _featVec(F._featVec)
    , _scaleLevels(F._scaleLevels)
    , _scaleFactor(F._scaleFactor)
    , _logScaleFactor(F._logScaleFactor)
    , _scaleFactors(F._scaleFactors)
    , _levelSigma2(F._levelSigma2)
    , _invLevelSigma2(F._invLevelSigma2)
    , _minX(int(F._minX))
    , _minY(int(F._minY))
    , _maxX(int(F._maxX))
    , _maxY(int(F._maxY))
    , _KFDB(pKFDB)
    , _vocabulary(F._vocabulary)
    , _notErase(false)
    , _toBeErased(false)
    , _bad(false)
    , _map(pMap)

{
    _camera.reset(F._camera->clone());

    _id = _nextId++;
    _node = make_shared<GraphNode>(this);

    _grid.resize(size_t(_gridCols));
    for (size_t i = 0; i < size_t(_gridCols); i++) {
        _grid[i].resize(size_t(_gridRows));
        for (size_t j = 0; j < size_t(_gridRows); j++)
            _grid[i][j] = F._grid[i][j];
    }

    _landmarks.resize(_features.size());
    for (size_t i = 0; i < _features.size(); ++i) {
        _landmarks[i] = _features[i]->_point;
    }

    setPose(F.getPose());
}

void KeyFrame::computeBoW()
{
    if (_bowVec.empty() || _featVec.empty()) {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(_descriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        _vocabulary->transform(vCurrentDesc, _bowVec, _featVec, 4);
    }
}

void KeyFrame::setPose(const SE3& Tcw)
{
    unique_lock<mutex> lock(_mutexPose);
    _Tcw = Tcw;
    _Twc = Tcw.inverse();
}

SE3 KeyFrame::getPose()
{
    unique_lock<mutex> lock(_mutexPose);
    return _Tcw;
}

SE3 KeyFrame::getPoseInverse()
{
    unique_lock<mutex> lock(_mutexPose);
    return _Twc;
}

void KeyFrame::addLandmark(LandmarkPtr pMP, const size_t& idx)
{
    unique_lock<mutex> lock(_mutexFeatures);
    _landmarks[idx] = pMP;
}

void KeyFrame::eraseLandmarkMatch(const size_t& idx)
{
    unique_lock<mutex> lock(_mutexFeatures);
    _landmarks[idx] = nullptr;
}

void KeyFrame::eraseLandmarkMatch(LandmarkPtr pMP)
{
    int idx = pMP->getIndexInKeyFrame(this);
    if (idx >= 0)
        _landmarks[size_t(idx)] = nullptr;
}

void KeyFrame::replaceLandmarkMatch(const size_t& idx, LandmarkPtr pMP)
{
    _landmarks[idx] = pMP;
}

set<LandmarkPtr> KeyFrame::getLandmarks()
{
    unique_lock<mutex> lock(_mutexFeatures);
    set<LandmarkPtr> s;
    for (LandmarkPtr pMP : _landmarks) {
        if (!pMP)
            continue;
        if (!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::trackedLandmarks(const int& minObs)
{
    unique_lock<mutex> lock(_mutexFeatures);

    int nPoints = 0;
    const bool bCheckObs = minObs > 0;
    for (size_t i = 0; i < _N; i++) {
        LandmarkPtr pMP = _landmarks[i];
        if (pMP) {
            if (!pMP->isBad()) {
                if (bCheckObs) {
                    if (_landmarks[i]->observations() >= minObs)
                        nPoints++;
                } else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<LandmarkPtr> KeyFrame::getLandmarkMatches()
{
    unique_lock<mutex> lock(_mutexFeatures);
    return _landmarks;
}

LandmarkPtr KeyFrame::getLandmark(const size_t& idx)
{
    unique_lock<mutex> lock(_mutexFeatures);
    return _landmarks[idx];
}

void KeyFrame::setNotErase()
{
    _notErase = true;
}

void KeyFrame::setErase()
{
    if (!_node->hasLoopEdge()) {
        _notErase = false;
    }

    if (_toBeErased) {
        setBadFlag();
    }
}

void KeyFrame::setBadFlag()
{
    if (_id == 0)
        return;
    else if (_notErase) {
        _toBeErased = true;
        return;
    }

    {
        unique_lock<mutex> lock1(_mutexFeatures);
        for (LandmarkPtr pMP : _landmarks) {
            if (pMP)
                pMP->eraseObservation(this);
        }
    }

    _node->eraseAllConnections();
    _node->recoverSpanningConnections();

    _Tcp = _Tcw * _node->getParent()->getPoseInverse();
    _bad = true;

    _map->eraseKeyFrame(this);
    _KFDB->erase(this);
}

bool KeyFrame::isBad()
{
    return _bad;
}

double KeyFrame::getDepth(const size_t& i)
{
    return _features[i]->_Xc.z();
}

vector<size_t> KeyFrame::getFeaturesInArea(const double& x, const double& y, const double& r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(_N);

    const int nMinCellX = max(0, static_cast<int>(floor((x - _minX - r) * _gridElementWidthInv)));
    if (nMinCellX >= _gridCols)
        return vIndices;

    const int nMaxCellX = min(static_cast<int>(_gridCols - 1), static_cast<int>(ceil((x - _minX + r) * _gridElementWidthInv)));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY = max(0, static_cast<int>(floor((y - _minY - r) * _gridElementHeightInv)));
    if (nMinCellY >= _gridRows)
        return vIndices;

    const int nMaxCellY = min(static_cast<int>(_gridRows - 1), static_cast<int>(ceil((y - _minY + r) * _gridElementHeightInv)));
    if (nMaxCellY < 0)
        return vIndices;

    for (size_t ix = size_t(nMinCellX); ix <= size_t(nMaxCellX); ix++) {
        for (size_t iy = size_t(nMinCellY); iy <= size_t(nMaxCellY); iy++) {
            const vector<size_t> vCell = _grid[ix][iy];
            for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                const FeaturePtr& kpUn = _features[vCell[j]];
                const double distx = kpUn->_uXi.x() - x;
                const double disty = kpUn->_uXi.y() - y;

                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::isInImage(const double& x, const double& y) const
{
    return (x >= _minX && x < _maxX && y >= _minY && y < _maxY);
}

void KeyFrame::createPointCloud(const int& res)
{
    if (hasPointCloud())
        return;

    _pointCloud.reset(new PointCloud());

    // Create point cloud in camera frame
    for (int m = 0; m < _depthIm.rows; m += res) {
        for (int n = 0; n < _depthIm.cols; n += res) {
            const double z = _depthIm.at<double>(m, n);

            if (z <= 0)
                continue;

            Vec3 xc = _camera->backproject(Vec2(n, m), z);
            PointT p;
            p.x = float(xc.x());
            p.y = float(xc.y());
            p.z = float(xc.z());

            p.b = _colorIm.data[m * _colorIm.step + n * _colorIm.channels() + 0]; // blue
            p.g = _colorIm.data[m * _colorIm.step + n * _colorIm.channels() + 1]; // green
            p.r = _colorIm.data[m * _colorIm.step + n * _colorIm.channels() + 2]; // red

            _pointCloud->points.push_back(p);
        }
    }

    _pointCloud->height = 1;
    _pointCloud->width = _pointCloud->points.size();
    _pointCloud->is_dense = false;
}

void KeyFrame::downsample(float leaf)
{
    if (!_pointCloud)
        return;

    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(leaf, leaf, leaf);

    voxel.setInputCloud(_pointCloud);
    voxel.filter(*_pointCloud);
}

void KeyFrame::statisticalFilter(int k, double stddev)
{
    if (!_pointCloud)
        return;

    pcl::StatisticalOutlierRemoval<PointT> sor;

    sor.setInputCloud(_pointCloud);
    sor.setMeanK(k);
    sor.setStddevMulThresh(stddev);
    sor.filter(*_pointCloud);
}

void KeyFrame::passThroughFilter(const string& field, float ll, float ul, const bool negate)
{
    if (!_pointCloud)
        return;

    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(_pointCloud);
    pass.setFilterFieldName(field);
    pass.setFilterLimits(ll, ul);
    if (negate)
        pass.setFilterLimitsNegative(true);

    pass.filter(*_pointCloud);
}

bool KeyFrame::hasPointCloud()
{
    return _pointCloud != nullptr;
}

void KeyFrame::createOctoCloud(PointCloud::Ptr worldCloud)
{
    _octoCloud.reset(new octomap::Pointcloud());
    _octoCloud->reserve(worldCloud->size());

    for (PointCloud::const_iterator it = worldCloud->begin(); it != worldCloud->end(); it++) {
        if (!isnan(it->z) || it->z <= 0)
            _octoCloud->push_back(it->x, it->y, it->z);
    }
}
