#ifndef SVOEXTRACTOR_H
#define SVOEXTRACTOR_H

#include <opencv2/features2d.hpp>
#include <vector>

class SVOextractor : public cv::Feature2D {
public:
    SVOextractor(int nLevels, int cellSize, double initThresh = 20.0,
                 int minKps = 600, int maxKps = 1000, double minThresh = 1.0, double maxThresh = 100.0,
                 double increaseFactor = 1.3, double decreaseFactor = 0.7);

    ~SVOextractor() {}

    CV_WRAP virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints,
        cv::InputArray mask = cv::noArray());

    void step(cv::Mat& image, std::vector<cv::KeyPoint>& kps);

    std::vector<cv::Mat> _imagePyramid;

protected:
    void createImagePyramid(cv::Mat& image);
    void resetGrid();
    void setExistingKeyPoints(std::vector<cv::KeyPoint>& keypoints);
    void setGridOccupancy(cv::Point2f& px);

    void tooFew();
    void tooMany();
    bool good() const;

    double _thresh;
    int _nLevels;
    int _cellSize;
    int _gridCols;
    int _gridRows;
    std::vector<bool> _gridOccupancy;

    // adaptive
    int _minKeyPoints;
    int _maxKeyPoints;
    double _minThresh;
    double _maxThresh;

    double _increaseFactor;
    double _decreaseFactor;
};

#endif // SVOEXTRACTOR_H
