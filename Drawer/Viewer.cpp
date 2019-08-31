#include "Viewer.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/Map.h"
#include "Drawer/DenseMapDrawer.h"
#include "MapDrawer.h"
#include "System/System.h"
#include "System/Tracking.h"
#include <mutex>
#include <opencv2/core.hpp>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;

Viewer::Viewer(System* pSystem, MapDrawerPtr pMapDrawer, TrackingPtr pTracking, MapPtr pMap,
    DenseMapPtr denseMap, const bool& start)
    : _system(pSystem)
    , _mapDrawer(pMapDrawer)
    , _tracker(pTracking)
    , _map(pMap)
    , _denseMap(denseMap)
    , _finishRequested(false)
    , _finished(true)
    , _stopped(true)
    , _stopRequested(false)
    , _meanTrackTime(0)
    , _rmseSac(0)
    , _scoreIcp(0)
{
    _imageWidth = 640;
    _imageHeight = 480;

    _viewpointX = 0;
    _viewpointY = -0.7f;
    _viewpointZ = -1.8f;
    _viewpointF = 500;

    if (start)
        _thread = thread(&Viewer::run, this);
}

void Viewer::start()
{
    if (!_thread.joinable())
        _thread = thread(&Viewer::run, this);
}

void Viewer::run()
{
    _finished = false;
    _stopped = false;

    pangolin::CreateWindowAndBind("Map Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowCurrentCamera("menu.CurrCam", true, true);
    pangolin::Var<bool> menuShowPoints("menu.Points", false, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.KeyFrames", true, true);
    pangolin::Var<bool> menuShowGraph("menu.Graph", true, true);
    pangolin::Var<double> menuTime("menu.Track time:", 0);
    pangolin::Var<int> menuNodes("menu.Nodes:", 0);
    pangolin::Var<int> menuPoints("menu.Map Points:", 0);
    pangolin::Var<int> menuMeanObservations("menu.Observations:", 0);
    pangolin::Var<int> menuMeanMatches("menu.Matches:", 0);
    pangolin::Var<int> menuMeanInliersTrack("menu.Inliers:", 0);
    pangolin::Var<double> menuMeanInlierRatio("menu.Inlier Ratio:", 0);
    pangolin::Var<int> menuLocalMPs("menu.Local MPs:", 0);
    pangolin::Var<int> menuLocalKFs("menu.Local KFs:", 0);
    pangolin::Var<double> menuRMSE("menu.RANSAC rmse: ", 0);
    pangolin::Var<double> menuScore("menu.ICP score:", 0);

    shared_ptr<pangolin::Var<bool>> menuShowDenseMap = nullptr;
    shared_ptr<pangolin::Var<bool>> menuSaveOctoMap = nullptr;
    shared_ptr<pangolin::Var<int>> menuSize = nullptr;
    shared_ptr<pangolin::Var<double>> menuMemory = nullptr;
    if (_denseMap) {
        menuShowDenseMap = make_shared<pangolin::Var<bool>>("menu.Dense Map", true, true);
        menuSize = make_shared<pangolin::Var<int>>("menu.Size", 0);
        menuMemory = make_shared<pangolin::Var<double>>("menu.Memory (Mb)", 0);
        menuSaveOctoMap = make_shared<pangolin::Var<bool>>("menu.Save Octomap", false, false);
    }

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, _viewpointF, _viewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(_viewpointX, _viewpointY, _viewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_features = pangolin::Display("imgFeatures")
                                     .SetAspect(/*1024.0*/ 2048.0 / 768.0);
    pangolin::GlTexture texFeatures(_imageWidth * 2, _imageHeight, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3f, pangolin::Attach::Pix(175), 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(d_features);

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    bool bFollow = true;

    while (1) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        _mapDrawer->getCurrentOpenGLCameraMatrix(Twc);

        if (menuFollowCamera && bFollow) {
            s_cam.Follow(Twc);
        } else if (menuFollowCamera && !bFollow) {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(_viewpointX, _viewpointY, _viewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        } else if (!menuFollowCamera && bFollow) {
            bFollow = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        if (menuShowCurrentCamera)
            _mapDrawer->drawCurrentCamera(Twc);
        if (menuShowKeyFrames || menuShowGraph)
            _mapDrawer->drawKeyFrames(menuShowKeyFrames, menuShowGraph);
        if (menuShowPoints)
            _mapDrawer->drawLandmarks();

        if (_denseMap) {
            if (menuShowDenseMap->Get())
                _denseMap->render();
            if (menuSaveOctoMap->Get()) {
                _denseMap->save("./OctoMap.ot");
                menuSaveOctoMap->Reset();
            }
            *menuSize = _denseMap->size();
            *menuMemory = double(_denseMap->memory()) / 1000000.0; // Mb
        }

        menuNodes = _map->KeyFramesInMap();
        menuPoints = _map->LandmarksInMap();
        menuMeanInliersTrack = _tracker->meanInliers();
        menuMeanInlierRatio = _tracker->meanInlierRatio();
        menuMeanObservations = _tracker->meanObservations();
        menuMeanMatches = _tracker->meanMatches();
        menuLocalKFs = _tracker->localKFs();
        menuLocalMPs = _tracker->localMPs();

        {
            unique_lock<mutex> lock(_mutexStatistics);
            menuTime = _meanTrackTime;
            menuRMSE = _rmseSac;
            menuScore = _scoreIcp;
        }

        cv::Mat feats = _tracker->getImageMatches();
        if (!feats.empty()) {
            texFeatures.Upload(feats.data, GL_BGR, GL_UNSIGNED_BYTE);
            d_features.Activate();
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            texFeatures.RenderToViewportFlipY();
        }

        pangolin::FinishFrame();

        if (stop()) {
            while (isStopped())
                usleep(3000);
        }

        if (checkFinish())
            break;
    }

    setFinish();
}

void Viewer::requestFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finishRequested = true;
}

bool Viewer::checkFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finishRequested;
}

void Viewer::setFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finished;
}

void Viewer::requestStop()
{
    unique_lock<mutex> lock(_mutexStop);
    if (!_stopped)
        _stopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(_mutexStop);
    return _stopped;
}

bool Viewer::stop()
{
    unique_lock<mutex> lock(_mutexStop);
    unique_lock<mutex> lock2(_mutexFinish);

    if (_finishRequested)
        return false;
    else if (_stopRequested) {
        _stopped = true;
        _stopRequested = false;
        return true;
    }

    return false;
}

void Viewer::release()
{
    unique_lock<mutex> lock(_mutexStop);
    _stopped = false;
}

void Viewer::join()
{
    if (_thread.joinable()) {
        _thread.join();
        cout << "Viewer thread JOINED" << endl;
    }
}

void Viewer::setMeanTime(const double& time)
{
    unique_lock<mutex> lock(_mutexStatistics);
    _meanTrackTime = time;
}

void Viewer::setRansacError(double rmse)
{
    unique_lock<mutex> lock(_mutexStatistics);
    _rmseSac = rmse;
}

void Viewer::setIcpScore(double score)
{
    unique_lock<mutex> lock(_mutexStatistics);
    _scoreIcp = score;
}
