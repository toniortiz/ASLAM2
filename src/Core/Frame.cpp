#include "Frame.h"
#include "Feature.h"
#include "Features/Extractor.h"
#include "KeyFrame.h"
#include "Landmark.h"
#include "PinholeCamera.h"
#include "System/Converter.h"
#include <opencv2/core/eigen.hpp>
#include <thread>

using namespace std;

int Frame::_nextId = 0;
bool Frame::_initialComputations = true;
double Frame::_minX, Frame::_minY, Frame::_maxX, Frame::_maxY;
double Frame::_gridElementWidthInv, Frame::_gridElementHeightInv;

Frame::Frame() {}

Frame::Frame(const cv::Mat& imBGR, const cv::Mat& imDepth, const double& timeStamp, ExtractorPtr extractor,
    VocabularyPtr voc, CameraPtr cam)
    : _vocabulary(voc)
    , _extractor(extractor)
    , _timestamp(timeStamp)
    , _camera(cam)
    , _colorIm(imBGR)
{
    _id = _nextId++;

    cvtColor(_colorIm, _grayIm, CV_BGR2GRAY);
    imDepth.convertTo(_depthIm, CV_64F, _camera->depthFactor());

    // Scale Level Info
    _scaleLevels = _extractor->getLevels();
    _scaleFactor = _extractor->getScaleFactor();
    _logScaleFactor = log(_scaleFactor);
    _scaleFactors = _extractor->getScaleFactors();
    _invScaleFactors = _extractor->getInverseScaleFactors();
    _levelSigma2 = _extractor->getScaleSigmaSquares();
    _invLevelSigma2 = _extractor->getInverseScaleSigmaSquares();

    // Feature extraction
    extract();

    _N = _keys.size();
    if (_keys.empty())
        return;

    _features.resize(_N);
    for (size_t i = 0; i < _N; ++i) {
        FeaturePtr ftr(new Feature(this, Vec2(_keys[i].pt.x, _keys[i].pt.y),
            _keys[i].octave, _keys[i].angle, _descriptors.row(i), i));
        _features[i] = ftr;
    }

    // This is done only for the first Frame (or after a change in the calibration)
    if (_initialComputations) {
        _camera->undistortBounds(_minX, _maxX, _minY, _maxY);

        _gridElementWidthInv = static_cast<double>(FRAME_GRID_COLS) / (_maxX - _minX);
        _gridElementHeightInv = static_cast<double>(FRAME_GRID_ROWS) / (_maxY - _minY);

        _initialComputations = false;
    }

    assignFeaturesToGrid();
}

void Frame::assignFeaturesToGrid()
{
    int nReserve = 0.5 * _N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            _grid[i][j].reserve(nReserve);

    for (size_t i = 0; i < _N; i++) {
        int nGridPosX, nGridPosY;
        if (posInGrid(_features[i]->_uXi, nGridPosX, nGridPosY))
            _grid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::extract()
{
    _extractor->detectAndCompute(_grayIm, cv::Mat(), _keys, _descriptors);
}

void Frame::setPose(const SE3& Tcw)
{
    _Tcw = Tcw;
    _Twc = Tcw.inverse();
}

SE3 Frame::getPose()
{
    return _Tcw;
}

SE3 Frame::getPoseInverse()
{
    return _Twc;
}

bool Frame::isInFrustum(LandmarkPtr pMP, double viewingCosLimit)
{
    pMP->_trackInView = false;

    // 3D in absolute coordinates
    Vec3 xw = pMP->getWorldPos();

    // 3D in camera coordinates
    Vec3 xc = _Tcw * xw;

    // Check positive depth
    if (xc.z() < 0.0)
        return false;

    // Project in image and check it is not outside
    Vec2 xp = _camera->project(xc);

    if (xp.x() < _minX || xp.x() > _maxX)
        return false;
    if (xp.y() < _minY || xp.y() > _maxY)
        return false;

    // Check distance is in the scale invariance region of the Landmark
    const double maxDistance = pMP->getMaxDistanceInvariance();
    const double minDistance = pMP->getMinDistanceInvariance();

    Vec3 PO = xw - _Twc.translation();

    const double dist = PO.norm();

    if (dist < minDistance || dist > maxDistance)
        return false;

    // Check viewing angle
    Vec3 Pn = pMP->getNormal();

    const double viewCos = PO.dot(Pn) / dist;

    if (viewCos < viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->predictScale(dist, this);

    // Data used by the tracking
    pMP->_trackInView = true;
    pMP->_trackProjX = xp.x();
    pMP->_trackProjXR = xp.x() - _camera->baseLineFx() * (1 / xc.z());
    pMP->_trackProjY = xp.y();
    pMP->_trackScaleLevel = nPredictedLevel;
    pMP->_trackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::getFeaturesInArea(const double& x, const double& y, const double& r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(_N);

    const int nMinCellX = max(0, (int)floor((x - _minX - r) * _gridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - _minX + r) * _gridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY = max(0, (int)floor((y - _minY - r) * _gridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - _minY + r) * _gridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
            const vector<size_t> vCell = _grid[ix][iy];
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                const FeaturePtr& ftr = _features[vCell[j]];
                if (bCheckLevels) {
                    if (ftr->_level < minLevel)
                        continue;
                    if (maxLevel >= 0)
                        if (ftr->_level > maxLevel)
                            continue;
                }

                const double distx = ftr->_uXi.x() - x;
                const double disty = ftr->_uXi.y() - y;

                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::posInGrid(const Vec2& kp, int& posX, int& posY)
{
    posX = round((kp.x() - _minX) * _gridElementWidthInv);
    posY = round((kp.y() - _minY) * _gridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

void Frame::computeBoW()
{
    if (_bowVec.empty()) {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(_descriptors);
        _vocabulary->transform(vCurrentDesc, _bowVec, _featVec, 4);
    }
}

double Frame::getDepth(const Vec2& xi)
{
    return _depthIm.at<double>(xi.y(), xi.x());
}

void Frame::addLandmark(const size_t& i, LandmarkPtr pMP) { _features[i]->_point = pMP; }

void Frame::eraseLandmark(const size_t& i) { _features[i]->_point = nullptr; }

void Frame::eraseLandmarks()
{
    for (FeaturePtr ftr : _features)
        ftr->_point = nullptr;
}

cv::Mat Frame::drawMatchedPoints()
{
    if (_keys.empty())
        return cv::Mat();

    cv::Mat out;
    cv::cvtColor(_grayIm, out, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < _N; i++) {
        LandmarkPtr pMP = _features[i]->_point;
        if (pMP) {
            if (_features[i]->isInlier()) {
                if (pMP->observations() > 0) {
                    const cv::KeyPoint& kp = _keys[i];
                    cv::circle(out, kp.pt, 4 * (kp.octave + 1), cv::Scalar(0, 255, 0), 1);
                }
            }
        }
    }

    return out;
}
