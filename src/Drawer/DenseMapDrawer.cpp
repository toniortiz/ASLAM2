#include "DenseMapDrawer.h"
#include "Core/KeyFrame.h"
#include "Core/Map.h"
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>

using namespace std;

DenseMapDrawer::DenseMapDrawer(MapPtr map, const bool& start)
    : _finishRequested(false)
    , _finished(true)
    , _map(map)
    , _octomap(0.05)
{
    _octomap.clear();
    _octomap.setClampingThresMin(0.001);
    _octomap.setClampingThresMax(0.999);
    _octomap.setResolution(0.05);
    _octomap.setOccupancyThres(0.5);
    _octomap.setProbHit(0.9);
    _octomap.setProbMiss(0.4);

    if (start)
        _thread = thread(&DenseMapDrawer::run, this);
}

void DenseMapDrawer::run()
{
    static int lastBigChange = 0;

    while (true) {
        vector<KeyFrame*> keyframes = _map->getAllKeyFrames();

        if (lastBigChange != _map->getLastBigChangeIdx()) {
            clear();
            for (KeyFrame* kf : keyframes) {
                if (kf->isBad())
                    continue;

                if (kf->hasPointCloud())
                    update(kf);
                else {
                    kf->createPointCloud(3);
                    kf->downsample(0.02f);
                    kf->passThroughFilter("z", 0.1f, 5.0f);
                    kf->statisticalFilter(50, 1.0);
                    update(kf);
                }
            }
            lastBigChange = _map->getLastBigChangeIdx();
            cout << "DenseMap UPDATED!" << endl;
        } else {
            for (KeyFrame* kf : keyframes) {
                if (kf->isBad())
                    continue;

                if (!kf->hasPointCloud()) {
                    kf->createPointCloud(3);
                    kf->downsample(0.02f);
                    kf->passThroughFilter("z", 0.1f, 5.0f);
                    kf->statisticalFilter(50, 1.0);

                    update(kf);
                }
            }
        }

        if (checkFinish())
            break;

        usleep(3000);
    }

    setFinish();
}

void DenseMapDrawer::update(KeyFrame* pKF)
{
    SE3 Twc = pKF->getPoseInverse();
    PointCloud::Ptr wc(new PointCloud);
    pcl::transformPointCloud(*pKF->_pointCloud, *wc, Twc.matrix().cast<float>());
    pKF->createOctoCloud(wc);

    Vec3 twc = Twc.translation();
    octomap::point3d origin(twc.x(), twc.y(), twc.z());

    unique_lock<mutex> lock(_mutexOctomap);
    _octomap.insertPointCloud(*pKF->_octoCloud, origin, -1, true);
    for (PointCloud::const_iterator it = wc->begin(); it != wc->end(); it++) {
        if (!isnan(it->x) && !isnan(it->y) && !isnan(it->z)) {
            const int rgb = *reinterpret_cast<const int*>(&(it->rgb));
            unsigned char r = ((rgb >> 16) & 0xff);
            unsigned char g = ((rgb >> 8) & 0xff);
            unsigned char b = (rgb & 0xff);
            _octomap.averageNodeColor(it->x, it->y, it->z, r, g, b);
        }
    }
    _octomap.updateInnerOccupancy();
}

void DenseMapDrawer::clear()
{
    unique_lock<mutex> lock(_mutexOctomap);
    _octomap.clear();
}

void DenseMapDrawer::save(const string& filename)
{
    ofstream ofile(filename, std::ios_base::out | std::ios_base::binary);
    if (!ofile.is_open())
        return;

    unique_lock<mutex> lock(_mutexOctomap);
    _octomap.write(ofile);
    ofile.close();

    cout << "Map SAVED!" << endl;
}

void DenseMapDrawer::render()
{
    unique_lock<mutex> lock(_mutexOctomap);

    octomap::ColorOcTree::tree_iterator it = _octomap.begin_tree();
    octomap::ColorOcTree::tree_iterator end = _octomap.end_tree();
    int counter = 0;
    double occThresh = 0.9;
    int level = 16;

    if (occThresh > 0) {
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBegin(GL_TRIANGLES);
        double stretch_factor = 128 / (1 - occThresh);

        for (; it != end; ++counter, ++it) {
            if (level != it.getDepth())
                continue;

            double occ = it->getOccupancy();
            if (occ < occThresh)
                continue;

            glColor4ub(it->getColor().r, it->getColor().g, it->getColor().b, 128 /*basic visibility*/ + (occ - occThresh) * stretch_factor);
            double halfsize = it.getSize() / 2.0;
            double x = it.getX();
            double y = it.getY();
            double z = it.getZ();
            //Front
            glVertex3d(x - halfsize, y - halfsize, z - halfsize);
            glVertex3d(x - halfsize, y + halfsize, z - halfsize);
            glVertex3d(x + halfsize, y + halfsize, z - halfsize);

            glVertex3d(x - halfsize, y - halfsize, z - halfsize);
            glVertex3d(x + halfsize, y + halfsize, z - halfsize);
            glVertex3d(x + halfsize, y - halfsize, z - halfsize);

            //Back
            glVertex3d(x - halfsize, y - halfsize, z + halfsize);
            glVertex3d(x + halfsize, y - halfsize, z + halfsize);
            glVertex3d(x + halfsize, y + halfsize, z + halfsize);

            glVertex3d(x - halfsize, y - halfsize, z + halfsize);
            glVertex3d(x + halfsize, y + halfsize, z + halfsize);
            glVertex3d(x - halfsize, y + halfsize, z + halfsize);

            //Left
            glVertex3d(x - halfsize, y - halfsize, z - halfsize);
            glVertex3d(x - halfsize, y - halfsize, z + halfsize);
            glVertex3d(x - halfsize, y + halfsize, z + halfsize);

            glVertex3d(x - halfsize, y - halfsize, z - halfsize);
            glVertex3d(x - halfsize, y + halfsize, z + halfsize);
            glVertex3d(x - halfsize, y + halfsize, z - halfsize);

            //Right
            glVertex3d(x + halfsize, y - halfsize, z - halfsize);
            glVertex3d(x + halfsize, y + halfsize, z - halfsize);
            glVertex3d(x + halfsize, y + halfsize, z + halfsize);

            glVertex3d(x + halfsize, y - halfsize, z - halfsize);
            glVertex3d(x + halfsize, y + halfsize, z + halfsize);
            glVertex3d(x + halfsize, y - halfsize, z + halfsize);

            //?
            glVertex3d(x - halfsize, y - halfsize, z - halfsize);
            glVertex3d(x + halfsize, y - halfsize, z - halfsize);
            glVertex3d(x + halfsize, y - halfsize, z + halfsize);

            glVertex3d(x - halfsize, y - halfsize, z - halfsize);
            glVertex3d(x + halfsize, y - halfsize, z + halfsize);
            glVertex3d(x - halfsize, y - halfsize, z + halfsize);

            //?
            glVertex3d(x - halfsize, y + halfsize, z - halfsize);
            glVertex3d(x - halfsize, y + halfsize, z + halfsize);
            glVertex3d(x + halfsize, y + halfsize, z + halfsize);

            glVertex3d(x - halfsize, y + halfsize, z - halfsize);
            glVertex3d(x + halfsize, y + halfsize, z + halfsize);
            glVertex3d(x + halfsize, y + halfsize, z - halfsize);
        }
        glEnd();
    }
}

void DenseMapDrawer::start()
{
    if (!_thread.joinable())
        _thread = thread(&DenseMapDrawer::run, this);
}

void DenseMapDrawer::join()
{
    if (_thread.joinable()) {
        _thread.join();
        cout << "DenseMapDrawer thread JOINED" << endl;
    }
}

void DenseMapDrawer::requestFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finishRequested = true;
}

bool DenseMapDrawer::isFinished()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finished;
}

int DenseMapDrawer::size()
{
    unique_lock<mutex> lock(_mutexOctomap);
    return int(_octomap.size());
}

int DenseMapDrawer::memory()
{
    unique_lock<mutex> lock(_mutexOctomap);
    return int(_octomap.memoryUsage());
}

double DenseMapDrawer::resolution()
{
    unique_lock<mutex> lock(_mutexOctomap);
    return _octomap.getResolution();
}

void DenseMapDrawer::setResolution(double res)
{
    unique_lock<mutex> lock(_mutexOctomap);
    _octomap.setResolution(res);
}

bool DenseMapDrawer::checkFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finishRequested;
}

void DenseMapDrawer::setFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finished = true;
}
