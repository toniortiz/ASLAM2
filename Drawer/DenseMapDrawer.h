#ifndef DENSEMAPDRAWER_H
#define DENSEMAPDRAWER_H

#include "System/Common.h"
#include <mutex>
#include <octomap/ColorOcTree.h>
#include <thread>

class DenseMapDrawer {
public:
    SMART_POINTER_TYPEDEFS(DenseMapDrawer);

    DenseMapDrawer(MapPtr map, const bool& start = true);

    void run();

    void save(const std::string& filename);

    void render();

    void start();
    void join();
    void requestFinish();
    bool isFinished();

    int size();
    int memory();

private:
    void update(KeyFrame* pKF);
    void clear();
    bool checkFinish();
    void setFinish();
    bool _finishRequested;
    bool _finished;
    std::mutex _mutexFinish;

    MapPtr _map;
    octomap::ColorOcTree _octomap;
    std::mutex _mutexOctomap;

    std::thread _thread;
};

#endif // DENSEMAPDRAWER_H
