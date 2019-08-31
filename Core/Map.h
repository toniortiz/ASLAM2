#ifndef MAP_H
#define MAP_H

#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>
#include "System/Common.h"


class Map {
public:
    SMART_POINTER_TYPEDEFS(Map);

    Map();

    void addKeyFrame(KeyFrame* pKF);
    void addLandmark(Landmark* pMP);
    void eraseLandmark(Landmark* pMP);
    void eraseLandmark(const int& id);
    void eraseKeyFrame(KeyFrame* pKF);
    void eraseKeyFrame(const int& id);
    void setReferenceLandmarks(const std::vector<Landmark*>& vpMPs);
    void informNewBigChange();
    int getLastBigChangeIdx();

    std::vector<KeyFrame*> getAllKeyFrames();
    std::vector<Landmark*> getAllLandmarks();
    std::vector<Landmark*> getReferenceLandmarks();

    size_t LandmarksInMap();
    size_t KeyFramesInMap();

    int getMaxKFid();

    void clear();

    std::vector<KeyFrame*> _keyFrameOrigins;

    std::mutex _mutexMapUpdate;

protected:
//    std::set<Landmark*> _landmarks;
//    std::set<KeyFrame*> _keyFrames;
    std::unordered_map<int, Landmark*> _landmarks;
    std::unordered_map<int, KeyFrame*> _keyframes;

    std::vector<Landmark*> _referenceLandmarks;

    int _maxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int _bigChangeIdx;

    std::mutex _mutexMap;
};

#endif // MAP_H
