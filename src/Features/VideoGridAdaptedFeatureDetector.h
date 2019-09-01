#ifndef VIDEOGRIDADAPTEDFEATUREDETECTOR_H
#define VIDEOGRIDADAPTEDFEATUREDETECTOR_H

#include "StatefulFeatureDetector.h"

class VideoGridAdaptedFeatureDetector : public StatefulFeatureDetector {
public:
    /**
     * \param detector            Detector that will be adapted.
     * \param maxTotalKeypoints   Maximum count of keypoints detected on the image. Only the strongest keypoints will be keeped.
     * \param gridRows            Grid rows count.
     * \param gridCols            Grid column count.
     * \param edgeThreshold       How much overlap is needed, to not lose keypoints at the inner borders of the grid (should be the same value, e.g., as edgeThreshold for ORB)
     */
    VideoGridAdaptedFeatureDetector(const cv::Ptr<StatefulFeatureDetector>& pDetector,
        int _maxTotalKeypoints = 1000, int _gridRows = 4, int _gridCols = 4, int _edgeThreshold = 31);

    virtual cv::Ptr<StatefulFeatureDetector> clone() const;

    CV_WRAP virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray());

protected:
    VideoGridAdaptedFeatureDetector& operator=(const VideoGridAdaptedFeatureDetector&);
    VideoGridAdaptedFeatureDetector(const VideoGridAdaptedFeatureDetector&);

    std::vector<cv::Ptr<StatefulFeatureDetector>> _detectors;
    int _maxTotalKeypoints;
    int _gridRows;
    int _gridCols;
    int _edgeThreshold;
};

#endif // VIDEOGRIDADAPTEDFEATUREDETECTOR_H
