#include "DatasetMicrosoft.h"
#include "Core/PinholeCamera.h"
#include <dirent.h>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>

using namespace std;

DatasetMicrosoft::DatasetMicrosoft()
    : Dataset("Microsoft")
{
    _camera = nullptr;
}

DatasetMicrosoft::~DatasetMicrosoft() {}

bool DatasetMicrosoft::isOpened() const
{
    return _camera != nullptr;
}

bool DatasetMicrosoft::open(const string& dataset)
{
    _camera.reset(new PinholeCamera(640, 480,
        585.0, 585.0, 320.0, 240.0,
        40.0, 40.0, 1000.0, 30,
        0, 0, 0, 0, 0));

    _baseDir = dataset;

    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(_baseDir.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string filename(ent->d_name);

            string::size_type idx = filename.find_last_of(".");
            string ext = filename.substr(idx + 1);

            if (ext != "png"s)
                continue;

            if (filename.find("depth") != string::npos)
                _imageFilenamesD.push_back(filename);
            else if (filename.find("color") != string::npos)
                _imageFilenamesRGB.push_back(filename);
        }
        closedir(dir);
    } else {
        cout << "Cannot open " << _baseDir << endl;
        return false;
    }

    sort(_imageFilenamesD.begin(), _imageFilenamesD.end());
    sort(_imageFilenamesRGB.begin(), _imageFilenamesRGB.end());

    _timestamps.resize(_imageFilenamesRGB.size());
    iota(_timestamps.begin(), _timestamps.end(), 0);
    // for_each(_timestamps.begin(), _timestamps.end(), [](const double& str) { cout << str << endl; });

    return true;
}

size_t DatasetMicrosoft::size() const
{
    return _imageFilenamesRGB.size();
}

pair<pair<cv::Mat, cv::Mat>, double> DatasetMicrosoft::getData(const size_t& i)
{
    cv::Mat imBGR = cv::imread(_baseDir + _imageFilenamesRGB[i], cv::IMREAD_COLOR);
    cv::Mat imD = cv::imread(_baseDir + _imageFilenamesD[i], cv::IMREAD_UNCHANGED);
    return { { imBGR, imD }, _timestamps[i] };
}

void DatasetMicrosoft::print(ostream& out, const string& text) const
{
    Dataset::print(out, text);
    out << "Base Dir: " << _baseDir << endl;
    out << "Size: " << size() << endl;
}

ostream& operator<<(ostream& out, const DatasetMicrosoft& dataset)
{
    dataset.print(out, string(""));
    return out;
}
