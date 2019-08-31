#include "LocalMap.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include "Core/GraphNode.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/PinholeCamera.h"
#include "Features/Matcher.h"
#include "System/Converter.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

typedef g2o::BlockSolver_6_3 BlockSolver;
typedef g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> CSLinearSolver;
typedef g2o::LinearSolverDense<BlockSolver::PoseMatrixType> DenseLinearSolver;
typedef g2o::OptimizationAlgorithmLevenberg LM;

using namespace std;

LocalMap::LocalMap(CameraPtr cam)
    : DELTA_MONO(sqrt(5.991))
    , DELTA_STEREO(sqrt(7.815))
{
    _optimizer = new g2o::SparseOptimizer();
    _optimizer->setAlgorithm(new LM(new BlockSolver(new DenseLinearSolver)));
    _optimizer->setVerbose(false);

    _camera.reset(cam->clone());
    _fx = _camera->fx();
    _fy = _camera->fy();
    _cx = _camera->cx();
    _cy = _camera->cy();
    _bf = _camera->baseLineFx();
}

LocalMap::~LocalMap()
{
    if (_optimizer)
        delete _optimizer;
}

bool LocalMap::track(FramePtr frame)
{
    _curFrame = frame;

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    searchLocalKFs();
    searchLocalPoints();

    projectLocalPoints();

    // Optimize Pose
    optimize();

    // Update Landmarks Statistics
    for (size_t i = 0; i < _curFrame->_N; i++) {
        if (_curFrame->_features[i]->_point) {
            if (_curFrame->_features[i]->isInlier())
                _curFrame->_features[i]->_point->increaseFound();
        }
    }

    release();
    return true;
}

void LocalMap::searchLocalKFs()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*, int> KFcounter;
    for (size_t i = 0; i < _curFrame->_N; i++) {
        if (_curFrame->_features[i]->_point) {
            Landmark* pMP = _curFrame->_features[i]->_point;
            if (!pMP->isBad()) {
                const map<KeyFrame*, size_t> observations = pMP->getObservations();
                for (auto& [pKFi, idx] : observations)
                    KFcounter[pKFi]++;
            } else {
                _curFrame->eraseLandmark(i);
            }
        }
    }

    if (KFcounter.empty())
        return;

    int max = 0;
    KeyFrame* pKFmax = nullptr;

    _localKFs.clear();
    _localKFs.reserve(3 * KFcounter.size());

    // All keyframes that observe a map point are included in the local map.
    // Also check which keyframe shares most points
    for (auto& [pKF, n] : KFcounter) {
        if (pKF->isBad())
            continue;

        if (n > max) {
            max = n;
            pKFmax = pKF;
        }

        _localKFs.push_back(pKF);
        pKF->_trackReferenceForFrame = _curFrame->_id;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (KeyFrame* pKF : _localKFs) {
        // Limit the number of keyframes
        if (_localKFs.size() > 80)
            break;

        const vector<KeyFrame*> vNeighs = pKF->_node->getBestNCovisibles(10);

        for (KeyFrame* pNeighKF : vNeighs) {
            if (!pNeighKF->isBad()) {
                if (pNeighKF->_trackReferenceForFrame != _curFrame->_id) {
                    _localKFs.push_back(pNeighKF);
                    pNeighKF->_trackReferenceForFrame = _curFrame->_id;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->_node->getChildrens();
        for (KeyFrame* pChildKF : spChilds) {
            if (!pChildKF->isBad()) {
                if (pChildKF->_trackReferenceForFrame != _curFrame->_id) {
                    _localKFs.push_back(pChildKF);
                    pChildKF->_trackReferenceForFrame = _curFrame->_id;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->_node->getParent();
        if (pParent) {
            if (pParent->_trackReferenceForFrame != _curFrame->_id) {
                _localKFs.push_back(pParent);
                pParent->_trackReferenceForFrame = _curFrame->_id;
                break;
            }
        }
    }

    if (pKFmax) {
        _refKF = pKFmax;
        _curFrame->_refKF = pKFmax;
    }
}

void LocalMap::searchLocalPoints()
{
    _localPoints.clear();

    for (KeyFrame* pKF : _localKFs) {
        const vector<Landmark*> vpMPs = pKF->getLandmarkMatches();

        for (Landmark* pMP : vpMPs) {
            if (!pMP)
                continue;
            if (pMP->_trackReferenceForFrame == _curFrame->_id)
                continue;
            if (!pMP->isBad()) {
                _localPoints.push_back(pMP);
                pMP->_trackReferenceForFrame = _curFrame->_id;
            }
        }
    }
}

void LocalMap::projectLocalPoints()
{
    // Do not search map points already matched
    for (size_t i = 0; i < _curFrame->_N; i++) {
        Landmark* pMP = _curFrame->_features[i]->_point;
        if (pMP) {
            if (pMP->isBad()) {
                pMP = nullptr;
            } else {
                pMP->increaseVisible();
                pMP->_lastFrameSeen = _curFrame->_id;
                pMP->_trackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (Landmark* pMP : _localPoints) {
        if (pMP->_lastFrameSeen == _curFrame->_id)
            continue;
        if (pMP->isBad())
            continue;
        // Project (this fills Landmark variables for matching)
        if (_curFrame->isInFrustum(pMP, 0.5)) {
            pMP->increaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0) {
        Matcher matcher(0.8f);
        int th = 3;
        matcher.searchByProjection(_curFrame, _localPoints, th);
    }
}

int LocalMap::optimize()
{
    int nInitialCorrespondences = 0;

    // Set Frame vertex
    g2o::VertexSE3Expmap* vSE3 = addVertex(_curFrame->getPose(), 0, false);

    // Set Landmark vertices
    const size_t N = _curFrame->_N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    {
        unique_lock<mutex> lock(Landmark::_globalMutex);

        for (size_t i = 0; i < N; i++) {
            Landmark* pMP = _curFrame->_features[i]->_point;
            if (pMP) {
                if (_curFrame->_features[i]->_right < 0) {
                    nInitialCorrespondences++;
                    _curFrame->_features[i]->setInlier();

                    const FeaturePtr kpUn = _curFrame->_features[i];
                    auto e = addMonoEdge(kpUn, pMP);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                } else {
                    nInitialCorrespondences++;
                    _curFrame->_features[i]->setInlier();
                    const FeaturePtr kpUn = _curFrame->_features[i];

                    auto e = addStereoEdge(kpUn, pMP);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
        }
    }

    if (nInitialCorrespondences < 3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const double chi2Mono[4] = { 5.991, 5.991, 5.991, 5.991 };
    const double chi2Stereo[4] = { 7.815, 7.815, 7.815, 7.815 };
    const int its[4] = { 10, 10, 10, 10 };

    int nBad = 0;
    for (size_t it = 0; it < 4; it++) {

        SE3 Tcw = _curFrame->getPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion(), Tcw.translation()));
        _optimizer->initializeOptimization(0);
        _optimizer->optimize(its[it]);

        nBad = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (_curFrame->_features[idx]->isOutlier()) {
                e->computeError();
            }

            const double chi2 = e->chi2();

            if (chi2 > chi2Mono[it]) {
                _curFrame->_features[idx]->setOutlier();
                e->setLevel(1);
                nBad++;
            } else {
                _curFrame->_features[idx]->setInlier();
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(nullptr);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (_curFrame->_features[idx]->isOutlier())
                e->computeError();

            const double chi2 = e->chi2();

            if (chi2 > chi2Stereo[it]) {
                _curFrame->_features[idx]->setOutlier();
                e->setLevel(1);
                nBad++;
            } else {
                e->setLevel(0);
                _curFrame->_features[idx]->setInlier();
            }

            if (it == 2)
                e->setRobustKernel(nullptr);
        }

        if (_optimizer->edges().size() < 10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(_optimizer->vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    SE3 pose(SE3quat_recov.rotation(), SE3quat_recov.translation());
    _curFrame->setPose(pose);

    return nInitialCorrespondences - nBad;
}

void LocalMap::release()
{
    if (_optimizer) {
        _optimizer->clear();
        delete _optimizer;
    }

    _optimizer = new g2o::SparseOptimizer();
    _optimizer->setAlgorithm(new LM(new BlockSolver(new DenseLinearSolver)));
    _optimizer->setVerbose(false);
}

g2o::VertexSE3Expmap* LocalMap::addVertex(const SE3& Tcw, int id, bool fixed)
{
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();

    vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion(), Tcw.translation()));
    vSE3->setId(id);
    vSE3->setFixed(fixed);

    _optimizer->addVertex(vSE3);

    return vSE3;
}

g2o::EdgeSE3ProjectXYZOnlyPose* LocalMap::addMonoEdge(const FeaturePtr kp, Landmark* pMP)
{
    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(_optimizer->vertex(0)));
    e->setMeasurement(kp->_uXi);
    const float invSigma2 = _curFrame->_invLevelSigma2[kp->_level];
    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(DELTA_MONO);

    e->fx = _fx;
    e->fy = _fy;
    e->cx = _cx;
    e->cy = _cy;
    Vec3 Xw = pMP->getWorldPos();
    e->Xw[0] = Xw.x();
    e->Xw[1] = Xw.y();
    e->Xw[2] = Xw.z();

    _optimizer->addEdge(e);
    return e;
}

g2o::EdgeStereoSE3ProjectXYZOnlyPose* LocalMap::addStereoEdge(const FeaturePtr kpUn, Landmark* pMP)
{
    Eigen::Matrix<double, 3, 1> obs;
    obs << kpUn->_uXi.x(), kpUn->_uXi.y(), kpUn->_right;

    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(_optimizer->vertex(0)));
    e->setMeasurement(obs);
    const float invSigma2 = _curFrame->_invLevelSigma2[kpUn->_level];
    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
    e->setInformation(Info);

    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(DELTA_STEREO);

    e->fx = _fx;
    e->fy = _fy;
    e->cx = _cx;
    e->cy = _cy;
    e->bf = _bf;
    Vec3 Xw = pMP->getWorldPos();
    e->Xw[0] = Xw.x();
    e->Xw[1] = Xw.y();
    e->Xw[2] = Xw.z();

    _optimizer->addEdge(e);
    return e;
}
