#include "DatasetICL.h"
#include "Core/PinholeCamera.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>

using namespace std;

DatasetICL::DatasetICL()
    : Dataset("ICL")
{
}

DatasetICL::~DatasetICL() {}

bool DatasetICL::isOpened() const { return (mAssociationFile.is_open()); }

pair<pair<cv::Mat, cv::Mat>, double> DatasetICL::getData(const size_t& i)
{
    cv::Mat imBGR = cv::imread(mBaseDir + mvImageFilenamesRGB[i], cv::IMREAD_COLOR);
    cv::Mat imD = cv::imread(mBaseDir + mvImageFilenamesD[i], cv::IMREAD_UNCHANGED);
    return { { imBGR, imD }, mvTimestamps[i] };
}

bool DatasetICL::open(const string& dataset)
{
    mBaseDir = dataset;
    mAssociationFile.open(mBaseDir + "associations.txt");
    if (!mAssociationFile.is_open()) {
        cerr << "Can't open association file" << endl;
        return false;
    }

    _camera.reset(new PinholeCamera(640, 480,
        481.2, 480.0, 319.5, 239.5,
        40.0, 40.0, 5000.0, 30.0,
        0, 0, 0, 0, 0));

    while (!mAssociationFile.eof()) {
        string s;
        getline(mAssociationFile, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            mvTimestamps.push_back(t);
            ss >> sRGB;
            mvImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            mvImageFilenamesD.push_back(sD);
        }
    }
    return true;
}

size_t DatasetICL::size() const { return mvImageFilenamesRGB.size(); }

void DatasetICL::print(ostream& out, const string& text) const
{
    Dataset::print(out, text);
    out << "Base Dir: " << mBaseDir << endl;
    out << "Size: " << size() << endl;
}

ostream& operator<<(ostream& out, const DatasetICL& dataset)
{
    dataset.print(out, string(""));
    return out;
}
