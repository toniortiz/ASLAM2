#ifndef DATASETAICL_H
#define DATASETAICL_H

#include "Dataset.h"

class DatasetAICL : public Dataset {
public:
    DatasetAICL();
    virtual ~DatasetAICL();

    bool open(const std::string& dataset) override;
    bool isOpened() const override;

    std::pair<std::pair<cv::Mat, cv::Mat>, double> getData(const size_t& i) override;

    size_t size() const override;

    void print(std::ostream& out, const std::string& text = "") const override;
    friend std::ostream& operator<<(std::ostream& out, const DatasetAICL& dataset);

protected:
    void readFiles(std::string directory, std::string ext, std::vector<std::string>* filenames);

    std::string _baseDir;

    std::vector<std::string> _imageFilenamesRGB;
    std::vector<std::string> _imageFilenamesD;
    std::vector<double> _timestamps;
};

#endif // DATASETAICL_H
