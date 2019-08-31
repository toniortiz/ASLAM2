#include "BundleAdjustment.h"
#include "Core/Feature.h"
#include "Core/GraphNode.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/Map.h"
#include "Core/PinholeCamera.h"
#include "System/Converter.h"
#include <Eigen/StdVector>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>

using namespace std;

void BundleAdjustment::Global(MapPtr pMap, int nIterations, bool* pbStopFlag, const int nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->getAllKeyFrames();
    vector<Landmark*> vpMP = pMap->getAllLandmarks();
    GBA(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

void BundleAdjustment::GBA(const vector<KeyFrame*>& vpKFs, const vector<Landmark*>& vpMPs,
    int nIterations, bool* pbStopFlag, const int nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMPs.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    int maxKFid = 0;

    // Set KeyFrame vertices
    for (KeyFrame* pKF : vpKFs) {
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(g2o::SE3Quat(pKF->getPose().unit_quaternion(), pKF->getPose().translation()));
        vSE3->setId(pKF->_id);
        vSE3->setFixed(pKF->_id == 0);
        optimizer.addVertex(vSE3);
        if (pKF->_id > maxKFid)
            maxKFid = pKF->_id;
    }

    const double thHuber2D = sqrt(5.99);
    const double thHuber3D = sqrt(7.815);

    // Set Landmark vertices
    for (size_t i = 0; i < vpMPs.size(); i++) {
        Landmark* pMP = vpMPs[i];
        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->getWorldPos());
        const int id = pMP->_id + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*, size_t> observations = pMP->getObservations();

        int nEdges = 0;
        //SET EDGES
        for (auto& [pKF, idx] : observations) {
            if (pKF->isBad() || pKF->_id > maxKFid)
                continue;

            nEdges++;

            const FeaturePtr kpUn = pKF->_features[idx];

            if (pKF->_features[idx]->_right < 0) {
                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->_id)));
                e->setMeasurement(kpUn->_uXi);
                const double& invSigma2 = pKF->_invLevelSigma2[kpUn->_level];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust) {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = static_cast<double>(pKF->_camera->fx());
                e->fy = static_cast<double>(pKF->_camera->fy());
                e->cx = static_cast<double>(pKF->_camera->cx());
                e->cy = static_cast<double>(pKF->_camera->cy());

                optimizer.addEdge(e);
            } else {
                Eigen::Matrix<double, 3, 1> obs;
                obs << kpUn->_uXi.x(), kpUn->_uXi.y(), kpUn->_right;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->_id)));
                e->setMeasurement(obs);
                const double& invSigma2 = pKF->_invLevelSigma2[kpUn->_level];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust) {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = static_cast<double>(pKF->_camera->fx());
                e->fy = static_cast<double>(pKF->_camera->fy());
                e->cx = static_cast<double>(pKF->_camera->cx());
                e->cy = static_cast<double>(pKF->_camera->cy());
                e->bf = static_cast<double>(pKF->_camera->baseLineFx());

                optimizer.addEdge(e);
            }
        }

        if (nEdges == 0) {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        } else {
            vbNotIncludedMP[i] = false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    // Keyframes
    for (KeyFrame* pKF : vpKFs) {
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->_id));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == 0) {
            pKF->setPose(SE3(SE3quat.rotation(), SE3quat.translation()));
        } else {
            pKF->_TcwGBA = SE3(SE3quat.rotation(), SE3quat.translation());
            pKF->_BAGlobalForKF = nLoopKF;
        }
    }

    // Points
    for (size_t i = 0; i < vpMPs.size(); i++) {
        if (vbNotIncludedMP[i])
            continue;

        Landmark* pMP = vpMPs[i];

        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->_id + maxKFid + 1));

        if (nLoopKF == 0) {
            pMP->setWorldPos(vPoint->estimate());
            pMP->updateNormalAndDepth();
        } else {
            pMP->_posGBA = vPoint->estimate();
            pMP->_BAGlobalForKF = nLoopKF;
        }
    }
}

void BundleAdjustment::Local(KeyFrame* pKF, bool* pbStopFlag, MapPtr pMap)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->_BALocalForKF = pKF->_id;

    const vector<KeyFrame*> vNeighKFs = pKF->_node->getCovisibles();
    for (KeyFrame* pKFi : vNeighKFs) {
        pKFi->_BALocalForKF = pKF->_id;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local Landmarks seen in Local KeyFrames
    list<Landmark*> lLocalLandmarks;
    for (KeyFrame* pLocalKF : lLocalKeyFrames) {
        vector<Landmark*> vpMPs = pLocalKF->getLandmarkMatches();
        for (Landmark* pMP : vpMPs) {
            if (pMP) {
                if (!pMP->isBad())
                    if (pMP->_BALocalForKF != pKF->_id) {
                        lLocalLandmarks.push_back(pMP);
                        pMP->_BALocalForKF = pKF->_id;
                    }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local Landmarks but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for (Landmark* pLocalMP : lLocalLandmarks) {
        map<KeyFrame*, size_t> observations = pLocalMP->getObservations();
        for (auto& [pKFi, idx] : observations) {
            if (pKFi->_BALocalForKF != pKF->_id && pKFi->_BAFixedForKF != pKF->_id) {
                pKFi->_BAFixedForKF = pKF->_id;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    int maxKFid = 0;

    // Set Local KeyFrame vertices
    for (KeyFrame* pKFi : lLocalKeyFrames) {
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(g2o::SE3Quat(pKFi->getPose().unit_quaternion(), pKFi->getPose().translation()));
        vSE3->setId(pKFi->_id);
        vSE3->setFixed(pKFi->_id == 0);
        optimizer.addVertex(vSE3);
        if (pKFi->_id > maxKFid)
            maxKFid = pKFi->_id;
    }

    // Set Fixed KeyFrame vertices
    for (KeyFrame* pKFi : lFixedCameras) {
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(g2o::SE3Quat(pKFi->getPose().unit_quaternion(), pKFi->getPose().translation()));
        vSE3->setId(pKFi->_id);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->_id > maxKFid)
            maxKFid = pKFi->_id;
    }

    // Set Landmark vertices
    const size_t nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalLandmarks.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<Landmark*> vpLandmarkEdgeMono;
    vpLandmarkEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<Landmark*> vpLandmarkEdgeStereo;
    vpLandmarkEdgeStereo.reserve(nExpectedSize);

    const double thHuberMono = sqrt(5.991);
    const double thHuberStereo = sqrt(7.815);

    for (Landmark* pMP : lLocalLandmarks) {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->getWorldPos());
        int id = pMP->_id + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*, size_t> observations = pMP->getObservations();

        // Set edges
        for (auto& [pKFi, idx] : observations) {
            if (!pKFi->isBad()) {
                const FeaturePtr kpUn = pKFi->_features[idx];

                // Monocular observation
                if (kpUn->_right < 0) {
                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->_id)));
                    e->setMeasurement(kpUn->_uXi);
                    const double& invSigma2 = pKFi->_invLevelSigma2[kpUn->_level];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = static_cast<double>(pKFi->_camera->fx());
                    e->fy = static_cast<double>(pKFi->_camera->fy());
                    e->cx = static_cast<double>(pKFi->_camera->cx());
                    e->cy = static_cast<double>(pKFi->_camera->cy());

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpLandmarkEdgeMono.push_back(pMP);
                }
                // Stereo observation
                else {
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn->_uXi.x(), kpUn->_uXi.y(), kpUn->_right;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->_id)));
                    e->setMeasurement(obs);
                    const double& invSigma2 = pKFi->_invLevelSigma2[kpUn->_level];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = static_cast<double>(pKFi->_camera->fx());
                    e->fy = static_cast<double>(pKFi->_camera->fy());
                    e->cx = static_cast<double>(pKFi->_camera->cx());
                    e->cy = static_cast<double>(pKFi->_camera->cy());
                    e->bf = static_cast<double>(pKFi->_camera->baseLineFx());

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpLandmarkEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore) {

        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            Landmark* pMP = vpLandmarkEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive()) {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            Landmark* pMP = vpLandmarkEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive()) {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Optimize again without the outliers

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame*, Landmark*>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        Landmark* pMP = vpLandmarkEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive()) {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        Landmark* pMP = vpLandmarkEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive()) {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->_mutexMapUpdate);

    if (!vToErase.empty()) {
        for (size_t i = 0; i < vToErase.size(); i++) {
            KeyFrame* pKFi = vToErase[i].first;
            Landmark* pMPi = vToErase[i].second;
            pKFi->eraseLandmarkMatch(pMPi);
            pMPi->eraseObservation(pKFi);
        }
    }

    // Recover optimized data

    // Keyframes
    for (KeyFrame* pLocalKF : lLocalKeyFrames) {
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pLocalKF->_id));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pLocalKF->setPose(SE3(SE3quat.rotation(), SE3quat.translation()));
    }

    // Points
    for (Landmark* pLocalMP : lLocalLandmarks) {
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pLocalMP->_id + maxKFid + 1));
        pLocalMP->setWorldPos(vPoint->estimate());
        pLocalMP->updateNormalAndDepth();
    }
}

int BundleAdjustment::OptimizeSE3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<Landmark*>& vpMatches1, g2o::Sim3& g2oS12, const double th2)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat& K1 = pKF1->_camera->_cvK;
    const cv::Mat& K2 = pKF2->_camera->_cvK;

    // Camera poses
    const Mat33 R1w = pKF1->getPose().rotationMatrix();
    const Vec3 t1w = pKF1->getPose().translation();
    const Mat33 R2w = pKF2->getPose().rotationMatrix();
    const Vec3 t2w = pKF2->getPose().translation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap* vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale = true;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<double>(0, 2);
    vSim3->_principle_point1[1] = K1.at<double>(1, 2);
    vSim3->_focal_length1[0] = K1.at<double>(0, 0);
    vSim3->_focal_length1[1] = K1.at<double>(1, 1);
    vSim3->_principle_point2[0] = K2.at<double>(0, 2);
    vSim3->_principle_point2[1] = K2.at<double>(1, 2);
    vSim3->_focal_length2[0] = K2.at<double>(0, 0);
    vSim3->_focal_length2[1] = K2.at<double>(1, 1);
    optimizer.addVertex(vSim3);

    // Set Landmark vertices
    const size_t N = vpMatches1.size();
    const vector<Landmark*> vpLandmarks1 = pKF1->getLandmarkMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2 * N);
    vpEdges12.reserve(2 * N);
    vpEdges21.reserve(2 * N);

    const double deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for (size_t i = 0; i < N; i++) {
        if (!vpMatches1[i])
            continue;

        Landmark* pMP1 = vpLandmarks1[i];
        Landmark* pMP2 = vpMatches1[i];

        const int id1 = 2 * i + 1;
        const int id2 = 2 * (i + 1);

        const int i2 = pMP2->getIndexInKeyFrame(pKF2);

        if (pMP1 && pMP2) {
            if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0) {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                Vec3 P3D1w = pMP1->getWorldPos();
                Vec3 P3D1c = R1w * P3D1w + t1w;
                vPoint1->setEstimate(P3D1c);
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Vec3 P3D2w = pMP2->getWorldPos();
                Vec3 P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(P3D2c);
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            } else
                continue;
        } else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        const FeaturePtr kpUn1 = pKF1->_features[i];

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(kpUn1->_uXi);
        const double& invSigmaSquare1 = pKF1->_invLevelSigma2[kpUn1->_level];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        const FeaturePtr kpUn2 = pKF2->_features[i2];

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(kpUn2->_uXi);
        double invSigmaSquare2 = pKF2->_invLevelSigma2[kpUn2->_level];
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++) {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = nullptr;
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ*>(nullptr);
            vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(nullptr);
            nBad++;
        }
    }

    int nMoreIterations;
    if (nBad > 0)
        nMoreIterations = 10;
    else
        nMoreIterations = 5;

    if (nCorrespondences - nBad < 10)
        return 0;

    // Optimize again only with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++) {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = nullptr;
        } else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();

    return nIn;
}
