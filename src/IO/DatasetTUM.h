#ifndef DATASETTUMRGBD_H
#define DATASETTUMRGBD_H

#include "Dataset.h"
#include <fstream>
#include <opencv2/core.hpp>
#include <vector>

class DatasetTUM : public Dataset {
public:
    DatasetTUM();
    virtual ~DatasetTUM();

    bool isOpened() const override;

    bool open(const std::string& dataset) override;
    size_t size() const override;

    void detectCamera();

    std::pair<std::pair<cv::Mat, cv::Mat>, double> getData(const size_t& i) override;

    void print(std::ostream& out, const std::string& text = "") const override;
    friend std::ostream& operator<<(std::ostream& out, const DatasetTUM& dataset);

protected:
    std::string _baseDir;
    std::ifstream _associationFile;

    std::vector<std::string> _imageFilenamesRGB;
    std::vector<std::string> _imageFilenamesD;
    std::vector<double> _timestamps;
};

#endif // DATASETTUMRGBD_H
