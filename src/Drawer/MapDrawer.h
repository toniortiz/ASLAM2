#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include "System/Common.h"
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <pangolin/pangolin.h>

class MapDrawer {
public:
    SMART_POINTER_TYPEDEFS(MapDrawer);

public:
    MapDrawer(MapPtr pMap);

    void drawLandmarks();
    void drawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    void drawCurrentCamera(pangolin::OpenGlMatrix& Twc);
    void setCurrentCameraPose(const SE3& Tcw);
    void setReferenceKeyFrame(KeyFrame* pKF);
    void getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M);

private:
    MapPtr _map;

    double _KFsize;
    float _KFlineWidth;
    float _graphLineWidth;
    float _pointSize;
    double _cameraSize;
    float _cameraLineWidth;

    SE3 _cameraPose;

    std::mutex _mutexCamera;
};

#endif // MAPDRAWER_H
