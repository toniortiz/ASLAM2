#ifndef VIDEODYNAMICADAPTEDFEATUREDETECTOR_H
#define VIDEODYNAMICADAPTEDFEATUREDETECTOR_H

#include "DetectorAdjuster.h"
#include "StatefulFeatureDetector.h"

class VideoDynamicAdaptedFeatureDetector : public StatefulFeatureDetector {
public:
    VideoDynamicAdaptedFeatureDetector(cv::Ptr<DetectorAdjuster> pAdjuster, int minFeatures = 400, int maxFeatures = 500, int maxIters = 5);

    virtual cv::Ptr<StatefulFeatureDetector> clone() const;

    CV_WRAP virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray());

private:
    VideoDynamicAdaptedFeatureDetector& operator=(const VideoDynamicAdaptedFeatureDetector&);
    VideoDynamicAdaptedFeatureDetector(const VideoDynamicAdaptedFeatureDetector&);

    int _escapeIters;
    int _minFeatures, _maxFeatures;
    mutable cv::Ptr<DetectorAdjuster> _adjuster;
};
#endif // VIDEODYNAMICADAPTEDFEATUREDETECTOR_H
