#ifndef DATASET_H
#define DATASET_H

#include "System/Common.h"
#include <memory>
#include <opencv2/core.hpp>
#include <string>

class Dataset {
public:
    SMART_POINTER_TYPEDEFS(Dataset);

    enum DS {
        TUM = 0,
        ICL,
        CORBS
    };

public:
    Dataset();
    Dataset(const std::string& name);
    virtual ~Dataset();

    std::string name() const;
    virtual bool isOpened() const;

    // First corresponds to color image and second with depthmap
    virtual std::pair<std::pair<cv::Mat, cv::Mat>, double> getData(const size_t& i) = 0;

    virtual size_t size() const;
    virtual bool open(const std::string& dataset);
    virtual void print(std::ostream& out, const std::string& text = "") const;

    static Dataset::Ptr create(const DS& dataset);

    std::ostream& operator<<(std::ostream& out);

    CameraPtr camera();

protected:
    std::string _name;
    CameraPtr _camera;
};

#endif // DATASET_H
