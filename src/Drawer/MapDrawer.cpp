#include "MapDrawer.h"
#include "Core/GraphNode.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/Map.h"
#include "System/System.h"
#include <mutex>
#include <pangolin/pangolin.h>

using namespace std;

MapDrawer::MapDrawer(MapPtr pMap)
    : _map(pMap)
    , _cameraPose(SE3(Mat44::Identity()))
{
    _KFsize = 0.05;
    _KFlineWidth = 1;
    _graphLineWidth = 0.9f;
    _pointSize = 2;
    _cameraSize = 0.07;
    _cameraLineWidth = 3;
}

void MapDrawer::drawLandmarks()
{
    const vector<LandmarkPtr>& vpMPs = _map->getAllLandmarks();
    const vector<LandmarkPtr>& vpRefMPs = _map->getReferenceLandmarks();

    set<LandmarkPtr> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if (vpMPs.empty())
        return;

    glPointSize(_pointSize);
    glBegin(GL_POINTS);
    glColor3d(0.25, 0.25, 0.25);

    for (LandmarkPtr pMP : vpMPs) {
        if (pMP->isBad() || spRefMPs.count(pMP))
            continue;
        Vec3 pos = pMP->getWorldPos();
        glVertex3d(pos.x(), pos.y(), pos.z());
    }
    glEnd();

    glPointSize(_pointSize);
    glBegin(GL_POINTS);
    glColor3d(0.8, 0.63, 0.18);

    for (LandmarkPtr pRefMP : spRefMPs) {
        if (pRefMP->isBad())
            continue;
        Vec3 pos = pRefMP->getWorldPos();
        glVertex3d(pos.x(), pos.y(), pos.z());
    }

    glEnd();
}

void MapDrawer::drawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const double& w = _KFsize;
    const double h = w * 0.75;
    const double z = w * 0.6;

    const vector<KeyFrame*> vpKFs = _map->getAllKeyFrames();

    if (bDrawKF) {
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame* pKF = vpKFs[i];
            Mat44 Twc = pKF->getPoseInverse().matrix();

            glPushMatrix();

            glMultMatrixd(Twc.data());

            glLineWidth(_KFlineWidth);
            glColor3d(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            glVertex3d(0, 0, 0);
            glVertex3d(w, h, z);
            glVertex3d(0, 0, 0);
            glVertex3d(w, -h, z);
            glVertex3d(0, 0, 0);
            glVertex3d(-w, -h, z);
            glVertex3d(0, 0, 0);
            glVertex3d(-w, h, z);

            glVertex3d(w, h, z);
            glVertex3d(w, -h, z);

            glVertex3d(-w, h, z);
            glVertex3d(-w, -h, z);

            glVertex3d(-w, h, z);
            glVertex3d(w, h, z);

            glVertex3d(-w, -h, z);
            glVertex3d(w, -h, z);
            glEnd();

            glPopMatrix();
        }
    }

    if (bDrawGraph) {
        glLineWidth(_graphLineWidth);
        glColor3d(0.0, 0.8, 0.8);
        glBegin(GL_LINES);

        for (KeyFrame* pKF : vpKFs) {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = pKF->_node->getCovisiblesByWeight(100);
            Vec3 Ow = pKF->getPoseInverse().translation();
            if (!vCovKFs.empty()) {
                for (KeyFrame* pCovKF : vCovKFs) {
                    if (pCovKF->_id < pKF->_id)
                        continue;
                    Vec3 Ow2 = pCovKF->getPoseInverse().translation();
                    glVertex3d(Ow.x(), Ow.y(), Ow.z());
                    glVertex3d(Ow2.x(), Ow2.y(), Ow2.z());
                }
            }

            // Spanning tree
            KeyFrame* pParent = pKF->_node->getParent();
            if (pParent) {
                Vec3 Owp = pParent->getPoseInverse().translation();
                glVertex3d(Ow.x(), Ow.y(), Ow.z());
                glVertex3d(Owp.x(), Owp.y(), Owp.z());
            }

            // Loops
            set<KeyFrame*> sLoopKFs = pKF->_node->getLoopEdges();
            for (KeyFrame* pLoopKF : sLoopKFs) {
                if (pLoopKF->_id < pKF->_id)
                    continue;
                Vec3 Owl = pLoopKF->getPoseInverse().translation();
                glVertex3d(Ow.x(), Ow.y(), Ow.z());
                glVertex3d(Owl.x(), Owl.y(), Owl.z());
            }
        }

        glEnd();
    }
}

void MapDrawer::drawCurrentCamera(pangolin::OpenGlMatrix& Twc)
{
    const double& w = _cameraSize;
    const double h = w * 0.75;
    const double z = w * 0.6;

    glPushMatrix();

    glMultMatrixd(Twc.m);

    glLineWidth(_cameraLineWidth);
    glColor3d(0.0, 0.0, 1.0);
    glBegin(GL_LINES);
    glVertex3d(0, 0, 0);
    glVertex3d(w, h, z);
    glVertex3d(0, 0, 0);
    glVertex3d(w, -h, z);
    glVertex3d(0, 0, 0);
    glVertex3d(-w, -h, z);
    glVertex3d(0, 0, 0);
    glVertex3d(-w, h, z);

    glVertex3d(w, h, z);
    glVertex3d(w, -h, z);

    glVertex3d(-w, h, z);
    glVertex3d(-w, -h, z);

    glVertex3d(-w, h, z);
    glVertex3d(w, h, z);

    glVertex3d(-w, -h, z);
    glVertex3d(w, -h, z);
    glEnd();

    glPopMatrix();
}

void MapDrawer::setCurrentCameraPose(const SE3& Tcw)
{
    unique_lock<mutex> lock(_mutexCamera);
    _cameraPose = Tcw;
}

void MapDrawer::getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M)
{
    if (!_cameraPose.matrix().isIdentity()) {
        Mat33 Rwc;
        Vec3 twc;
        {
            unique_lock<mutex> lock(_mutexCamera);
            Rwc = _cameraPose.rotationMatrix().transpose();
            twc = -Rwc * _cameraPose.translation();
        }

        M.m[0] = static_cast<pangolin::GLprecision>(Rwc(0, 0));
        M.m[1] = static_cast<pangolin::GLprecision>(Rwc(1, 0));
        M.m[2] = static_cast<pangolin::GLprecision>(Rwc(2, 0));
        M.m[3] = 0.0;

        M.m[4] = static_cast<pangolin::GLprecision>(Rwc(0, 1));
        M.m[5] = static_cast<pangolin::GLprecision>(Rwc(1, 1));
        M.m[6] = static_cast<pangolin::GLprecision>(Rwc(2, 1));
        M.m[7] = 0.0;

        M.m[8] = static_cast<pangolin::GLprecision>(Rwc(0, 2));
        M.m[9] = static_cast<pangolin::GLprecision>(Rwc(1, 2));
        M.m[10] = static_cast<pangolin::GLprecision>(Rwc(2, 2));
        M.m[11] = 0.0;

        M.m[12] = static_cast<pangolin::GLprecision>(twc.x());
        M.m[13] = static_cast<pangolin::GLprecision>(twc.y());
        M.m[14] = static_cast<pangolin::GLprecision>(twc.z());
        M.m[15] = 1.0;
    } else
        M.SetIdentity();
}
