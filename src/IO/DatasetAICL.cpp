#include "DatasetAICL.h"
#include "Core/PinholeCamera.h"
#include <dirent.h>
#include <opencv2/highgui.hpp>
#include <thread>

using namespace std;

DatasetAICL::DatasetAICL()
    : Dataset("A-ICL")
{
    _camera = nullptr;
}

DatasetAICL::~DatasetAICL() {}

bool DatasetAICL::open(const string& dataset)
{
    _baseDir = dataset;

    _camera.reset(new PinholeCamera(640, 480,
        481.2, 480.0, 319.5, 239.5,
        40.0, 40.0, 1000.0, 30.0,
        0, 0, 0, 0, 0));

    thread t1(&DatasetAICL::readFiles, this, _baseDir + "color/", "jpg", &_imageFilenamesRGB);
    thread t2(&DatasetAICL::readFiles, this, _baseDir + "depth/", "png", &_imageFilenamesD);
    t1.join();
    t2.join();

    _timestamps.resize(_imageFilenamesRGB.size());
    iota(_timestamps.begin(), _timestamps.end(), 0);

    return true;
}

bool DatasetAICL::isOpened() const
{
    return _camera != nullptr;
}

pair<pair<cv::Mat, cv::Mat>, double> DatasetAICL::getData(const size_t& i)
{
    cv::Mat imBGR = cv::imread(_baseDir + "color/" + _imageFilenamesRGB[i], cv::IMREAD_COLOR);
    cv::Mat imD = cv::imread(_baseDir + "depth/" + _imageFilenamesD[i], cv::IMREAD_UNCHANGED);
    return { { imBGR, imD }, _timestamps[i] };
}

size_t DatasetAICL::size() const
{
    return _imageFilenamesRGB.size();
}

void DatasetAICL::print(ostream& out, const string& text) const
{
    Dataset::print(out, text);
    out << "Base Dir: " << _baseDir << endl;
    out << "Size: " << size() << endl;
}

void DatasetAICL::readFiles(string directory, string ext, vector<string>* filenames)
{
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string filename(ent->d_name);

            string::size_type idx = filename.find_last_of(".");

            if (filename.substr(idx + 1) != ext)
                continue;

            filenames->push_back(filename);
        }
        closedir(dir);
    } else {
        cout << "Cannot open " << directory << endl;
    }

    sort(filenames->begin(), filenames->end());
}

ostream& operator<<(ostream& out, const DatasetAICL& dataset)
{
    dataset.print(out, string(""));
    return out;
}
