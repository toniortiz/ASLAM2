#ifndef DATASETICLRGBD_H
#define DATASETICLRGBD_H

#include "Dataset.h"
#include <fstream>
#include <vector>

class DatasetICL : public Dataset {
public:
    DatasetICL();
    virtual ~DatasetICL();

    bool isOpened() const override;
    std::pair<std::pair<cv::Mat, cv::Mat>, double> getData(const size_t& i) override;
    bool open(const std::string& dataset) override;
    size_t size() const override;

    void print(std::ostream& out, const std::string& text = "") const override;
    friend std::ostream& operator<<(std::ostream& out, const DatasetICL& dataset);

protected:
    std::string mBaseDir;
    std::ifstream mAssociationFile;

    std::vector<std::string> mvImageFilenamesRGB;
    std::vector<std::string> mvImageFilenamesD;
    std::vector<double> mvTimestamps;
};

#endif // DATASETICLRGBD_H
