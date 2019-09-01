#include "VideoDynamicAdaptedFeatureDetector.h"

using namespace std;

VideoDynamicAdaptedFeatureDetector::VideoDynamicAdaptedFeatureDetector(cv::Ptr<DetectorAdjuster> pAdjuster,
    int minFeatures, int maxFeatures, int maxIters)
    : _escapeIters(maxIters)
    , _minFeatures(minFeatures)
    , _maxFeatures(maxFeatures)
    , _adjuster(pAdjuster)
{
}

cv::Ptr<StatefulFeatureDetector> VideoDynamicAdaptedFeatureDetector::clone() const
{
    StatefulFeatureDetector* pNew = new VideoDynamicAdaptedFeatureDetector(_adjuster->clone(),
        _minFeatures,
        _maxFeatures,
        _escapeIters);
    cv::Ptr<StatefulFeatureDetector> pCloned(pNew);
    return pCloned;
}

void VideoDynamicAdaptedFeatureDetector::detect(cv::InputArray image, vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    int iterCount = _escapeIters;

    do {
        keypoints.clear();

        _adjuster->detect(image, keypoints, mask);
        int keypointsFound = static_cast<int>(keypoints.size());

        if (keypointsFound < _minFeatures) {
            _adjuster->tooFew(_minFeatures, keypointsFound);
        } else if (int(keypoints.size()) > _maxFeatures) {
            _adjuster->tooMany(_maxFeatures, (int)keypoints.size());
            break;
        } else
            break;

        iterCount--;
    } while (iterCount > 0 && _adjuster->good());
}
