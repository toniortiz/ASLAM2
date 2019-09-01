#include "Dataset.h"
#include "DatasetCORBS.h"
#include "DatasetICL.h"
#include "DatasetTUM.h"

using namespace std;

Dataset::Dataset()
    : _name("Untitled")
{
}

Dataset::Dataset(const string& name)
    : _name(name)
{
}

Dataset::~Dataset() {}

string Dataset::name() const { return _name; }

bool Dataset::isOpened() const { return false; }

size_t Dataset::size() const { return 0; }

bool Dataset::open(const string& dataset) { return false; }

Dataset::Ptr Dataset::create(const DS& dataset)
{
    switch (dataset) {
    case TUM:
        return make_shared<DatasetTUM>();
    case ICL:
        return make_shared<DatasetICL>();
    case CORBS:
        return make_shared<DatasetCORBS>();
    }

    return nullptr;
}

void Dataset::print(ostream& out, const string& text) const
{
    if (text.size() > 0)
        out << text << endl;

    out << "Name: " << _name << endl;
}

ostream& Dataset::operator<<(ostream& out)
{
    print(out, string(""));
    return out;
}

CameraPtr Dataset::camera()
{
    return _camera;
}
