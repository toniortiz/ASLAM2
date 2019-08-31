#ifndef DATASETMICROSOFT_H
#define DATASETMICROSOFT_H

#include "Dataset.h"
#include "System/Common.h"

class DatasetMicrosoft : public Dataset {
public:
    SMART_POINTER_TYPEDEFS(DatasetMicrosoft);

    DatasetMicrosoft();

    virtual ~DatasetMicrosoft();

    bool isOpened() const override;

    bool open(const std::string& dataset) override;

    size_t size() const override;

    std::pair<std::pair<cv::Mat, cv::Mat>, double> getData(const size_t& i) override;

    void print(std::ostream& out, const std::string& text = "") const override;
    friend std::ostream& operator<<(std::ostream& out, const DatasetMicrosoft& dataset);

protected:
    std::string _baseDir;

    std::vector<std::string> _imageFilenamesRGB;
    std::vector<std::string> _imageFilenamesD;
    std::vector<double> _timestamps;
};

#endif // DATASETMICROSOFT_H
