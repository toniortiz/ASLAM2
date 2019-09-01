#include "DatasetTUM.h"
#include "Core/PinholeCamera.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>

using namespace std;

DatasetTUM::DatasetTUM()
    : Dataset("TUM")
{
}

DatasetTUM::~DatasetTUM() {}

bool DatasetTUM::isOpened() const { return (_associationFile.is_open()); }

bool DatasetTUM::open(const string& dataset)
{
    _baseDir = dataset;
    _associationFile.open(_baseDir + "associations.txt");
    if (!_associationFile.is_open()) {
        cerr << "Can't open association file" << endl;
        return false;
    }

    detectCamera();

    while (!_associationFile.eof()) {
        string s;
        getline(_associationFile, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            _timestamps.push_back(t);
            ss >> sRGB;
            _imageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            _imageFilenamesD.push_back(sD);
        }
    }
    return true;
}

size_t DatasetTUM::size() const { return _imageFilenamesRGB.size(); }

pair<pair<cv::Mat, cv::Mat>, double> DatasetTUM::getData(const size_t& i)
{
    cv::Mat imBGR = cv::imread(_baseDir + _imageFilenamesRGB[i], cv::IMREAD_COLOR);
    cv::Mat imD = cv::imread(_baseDir + _imageFilenamesD[i], cv::IMREAD_UNCHANGED);
    return { { imBGR, imD }, _timestamps[i] };
}

void DatasetTUM::detectCamera()
{
    string::size_type idx = _baseDir.find("freiburg");

    char c = _baseDir.at(idx + 8);
    switch (c) {
    case '1':
        _camera.reset(new PinholeCamera(640, 480,
            517.306408, 516.469215, 318.643040, 255.313989,
            40.0, 40.0, 5000.0, 30.0,
            0.262383, -0.953104, 1.163314, -0.005358, 0.002628));
        break;

    case '2':
        _camera.reset(new PinholeCamera(640, 480,
            520.908620, 521.007327, 325.141442, 249.701764,
            40.0, 40.0, 5208.0, 30.0,
            0.231222, -0.784899, 0.917205, -0.003257, -0.000105));
        break;

    case '3':
        _camera.reset(new PinholeCamera(640, 480,
            535.4, 539.2, 320.1, 247.6,
            40.0, 40.0, 5000.0, 30.0,
            0, 0, 0, 0, 0));
        break;
    }
}

void DatasetTUM::print(ostream& out, const string& text) const
{
    Dataset::print(out, text);
    out << "Base Dir: " << _baseDir << endl;
    out << "Size: " << size() << endl;
}

ostream& operator<<(ostream& out, const DatasetTUM& dataset)
{
    dataset.print(out, string(""));
    return out;
}
