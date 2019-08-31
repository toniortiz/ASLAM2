#include "SVOextractor.h"
#include <fast/fast.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

void halfSample(const cv::Mat& in, cv::Mat& out)
{
    assert(in.rows / 2 == out.rows && in.cols / 2 == out.cols);
    assert(in.type() == CV_8UC1 && out.type() == CV_8UC1);

    const int stride = in.step.p[0];
    uint8_t* top = (uint8_t*)in.data;
    uint8_t* bottom = top + stride;
    uint8_t* end = top + stride * in.rows;
    const int out_width = out.cols;
    uint8_t* p = (uint8_t*)out.data;
    while (bottom < end) {
        for (int j = 0; j < out_width; j++) {
            *p = static_cast<uint8_t>((uint16_t(top[0]) + top[1] + bottom[0] + bottom[1]) / 4);
            p++;
            top += 2;
            bottom += 2;
        }
        top += stride;
        bottom += stride;
    }
}

float ShiTomasiScore(const cv::Mat& img, int u, int v)
{
    assert(img.type() == CV_8UC1);

    float dXX = 0.0;
    float dYY = 0.0;
    float dXY = 0.0;
    const int halfbox_size = 4;
    const int box_size = 2 * halfbox_size;
    const int box_area = box_size * box_size;
    const int x_min = u - halfbox_size;
    const int x_max = u + halfbox_size;
    const int y_min = v - halfbox_size;
    const int y_max = v + halfbox_size;

    if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
        return 0.0; // patch is too close to the boundary

    const int stride = img.step.p[0];
    for (int y = y_min; y < y_max; ++y) {
        const uint8_t* ptr_left = img.data + stride * y + x_min - 1;
        const uint8_t* ptr_right = img.data + stride * y + x_min + 1;
        const uint8_t* ptr_top = img.data + stride * (y - 1) + x_min;
        const uint8_t* ptr_bottom = img.data + stride * (y + 1) + x_min;
        for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
            float dx = *ptr_right - *ptr_left;
            float dy = *ptr_bottom - *ptr_top;
            dXX += dx * dx;
            dYY += dy * dy;
            dXY += dx * dy;
        }
    }

    // Find and return smaller eigenvalue:
    dXX = dXX / (2.0 * box_area);
    dYY = dYY / (2.0 * box_area);
    dXY = dXY / (2.0 * box_area);
    return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}

SVOextractor::SVOextractor(int nLevels, int cellSize, double initThresh, int minKps, int maxKps,
    double minThresh, double maxThresh, double increaseFactor, double decreaseFactor)
    : _nLevels(nLevels)
    , _cellSize(cellSize)
    , _thresh(initThresh)
    , _minKeyPoints(minKps)
    , _maxKeyPoints(maxKps)
    , _minThresh(minThresh)
    , _maxThresh(maxThresh)
    , _increaseFactor(increaseFactor)
    , _decreaseFactor(decreaseFactor)
{
}

void SVOextractor::detect(cv::InputArray _image, vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    if (_image.empty())
        return;
    cv::Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);

    _gridCols = ceil(static_cast<double>(image.cols) / _cellSize);
    _gridRows = ceil(static_cast<double>(image.rows) / _cellSize);
    _gridOccupancy = vector<bool>(_gridCols * _gridRows, false);

    createImagePyramid(image);

    int iters = 5;
    vector<cv::KeyPoint> kps;
    do {
        kps.clear();
        step(image, kps);
        int kpsFound = kps.size();

        if (kpsFound < _minKeyPoints) {
            tooFew();
        } else if (kpsFound > _maxKeyPoints) {
            tooMany();
            break;
        } else {
            break;
        }
        iters--;

    } while (iters > 0 && good());

    keypoints = kps;
}

void SVOextractor::step(cv::Mat& image, vector<cv::KeyPoint>& kps)
{
    vector<cv::KeyPoint> _keypoints(_gridCols * _gridRows, cv::KeyPoint(0, 0, 0, -1, _thresh, 0, -1));

    /*
    for (int L = 0; L < _nLevels; ++L) {
        const int scale = (1 << L);
        vector<cv::KeyPoint> levelKps;
        cv::FAST(pyr[L], levelKps, _thresh, true);

    for (const auto& corner : levelKps) {
        const int k = static_cast<int>((corner.pt.y * scale) / _cellSize) * _gridCols + static_cast<int>((corner.pt.x * scale) / _cellSize);
        if (_gridOccupancy[k])
            continue;

        if (corner.response > keypoints.at(k).response) {
            cv::KeyPoint kp;
            kp.pt = cv::Point2f(corner.pt.x * scale, corner.pt.y * scale);
            kp.response = corner.response;
            kp.octave = L;
            keypoints.at(k) = kp;
        }
    }
}
*/

    for (int L = 0; L < _nLevels; L++) {
        const int scale = (1 << L);

        cv::Mat& workingMat = _imagePyramid[L];

        vector<fast::fast_xy> vFastCorners;
        fast::fast_corner_detect_10((fast::fast_byte*)workingMat.data, workingMat.cols,
            workingMat.rows, workingMat.cols, 20, vFastCorners);

        vector<int> vScores;
        vector<int> vnmCorners;
        fast::fast_corner_score_10((fast::fast_byte*)workingMat.data, workingMat.cols,
            vFastCorners, 20, vScores);
        fast::fast_nonmax_3x3(vFastCorners, vScores, vnmCorners);

        for (auto it = vnmCorners.begin(), ite = vnmCorners.end(); it != ite; ++it) {
            fast::fast_xy& pt = vFastCorners.at(*it);
            const int k = static_cast<int>((pt.y * scale) / _cellSize) * _gridCols + static_cast<int>((pt.x * scale) / _cellSize);

            if (_gridOccupancy[k])
                continue;
            const float score = ShiTomasiScore(workingMat, pt.x, pt.y);
            if (score > _keypoints.at(k).response) {
                cv::KeyPoint kp;
                kp.pt = cv::Point2f(pt.x * scale, pt.y * scale);
                kp.response = score;
                kp.octave = L;
                _keypoints.at(k) = kp;
            }
        }
    }

    for (auto& kp : _keypoints) {
        if (kp.response > _thresh)
            kps.push_back(kp);
    }

    resetGrid();
}

void SVOextractor::createImagePyramid(cv::Mat& image)
{
    _imagePyramid.clear();
    _imagePyramid.resize(_nLevels);
    _imagePyramid[0] = image;
    for (int i = 1; i < _nLevels; ++i) {
        _imagePyramid[i] = cv::Mat(_imagePyramid[i - 1].rows / 2, _imagePyramid[i - 1].cols / 2, CV_8UC1);
        halfSample(_imagePyramid[i - 1], _imagePyramid[i]);
    }
}

void SVOextractor::resetGrid()
{
    fill(_gridOccupancy.begin(), _gridOccupancy.end(), false);
}

void SVOextractor::setExistingKeyPoints(std::vector<cv::KeyPoint>& keypoints)
{
    for_each(keypoints.begin(), keypoints.end(), [&](cv::KeyPoint& kp) {
        _gridOccupancy.at(
            static_cast<int>(kp.pt.y / _cellSize) * _gridCols
            + static_cast<int>(kp.pt.x / _cellSize))
            = true;
    });
}

void SVOextractor::setGridOccupancy(cv::Point2f& px)
{
    _gridOccupancy.at(
        static_cast<int>(px.y / _cellSize) * _gridCols
        + static_cast<int>(px.x / _cellSize))
        = true;
}

void SVOextractor::tooFew()
{
    _thresh *= _decreaseFactor;
    if (_thresh < _minThresh)
        _thresh = _minThresh;
}

void SVOextractor::tooMany()
{
    _thresh *= _increaseFactor;
    if (_thresh > _maxThresh)
        _thresh = _maxThresh;
}

bool SVOextractor::good() const
{
    return (_thresh > _minThresh) && (_thresh < _maxThresh);
}
