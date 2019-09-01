#include "PoseGraph.h"
#include "Core/GraphNode.h"
#include "Core/KeyFrame.h"
#include "Core/Landmark.h"
#include "Core/Map.h"
#include "System/Common.h"
#include "System/Converter.h"
#include <Eigen/StdVector>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>

using namespace std;

void PoseGraph::optimize(MapPtr pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF, const LoopClosing::KeyFrameAndPose& NonCorrectedSE3,
    const LoopClosing::KeyFrameAndPose& CorrectedSE3, const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->getAllKeyFrames();
    const vector<LandmarkPtr> vpMPs = pMap->getAllLandmarks();

    const int nMaxKFid = pMap->getMaxKFid();

    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vScw(nMaxKFid + 1);
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vCorrectedSwc(nMaxKFid + 1);
    vector<g2o::VertexSE3Expmap*> vpVertices(nMaxKFid + 1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();

        const int nIDi = pKF->_id;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSE3.find(pKF);

        if (it != CorrectedSE3.end()) {
            vScw[nIDi] = it->second;
            vSE3->setEstimate(it->second);
        } else {
            Mat33 Rcw = pKF->getPose().rotationMatrix();
            Vec3 tcw = pKF->getPose().translation();
            g2o::SE3Quat Siw(Rcw, tcw);
            vScw[nIDi] = Siw;
            vSE3->setEstimate(Siw);
        }

        if (pKF == pLoopKF)
            vSE3->setFixed(true);

        vSE3->setId(nIDi);
        vSE3->setMarginalized(false);

        optimizer.addVertex(vSE3);

        vpVertices[nIDi] = vSE3;
    }

    set<pair<int, int>> sInsertedEdges;

    const Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();

    // Set Loop edges
    for (auto& [pKF, spConnections] : LoopConnections) {
        const int nIDi = pKF->_id;
        const g2o::SE3Quat Siw = vScw[nIDi];
        const g2o::SE3Quat Swi = Siw.inverse();

        for (KeyFrame* pLoopKF : spConnections) {
            const int nIDj = pLoopKF->_id;
            if ((nIDi != pCurKF->_id || nIDj != pLoopKF->_id) && pKF->_node->getWeight(pLoopKF) < minFeat)
                continue;

            const g2o::SE3Quat Sjw = vScw[nIDj];
            const g2o::SE3Quat Sji = Sjw * Swi;

            g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    // Set normal edges
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->_id;

        g2o::SE3Quat Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSE3.find(pKF);

        if (iti != NonCorrectedSE3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->_node->getParent();

        // Spanning tree edge
        if (pParentKF) {
            int nIDj = pParentKF->_id;

            g2o::SE3Quat Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSE3.find(pParentKF);

            if (itj != NonCorrectedSE3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::SE3Quat Sji = Sjw * Swi;

            g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->_node->getLoopEdges();
        for (KeyFrame* pLKF : sLoopEdges) {
            if (pLKF->_id < pKF->_id) {
                g2o::SE3Quat Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSE3.find(pLKF);

                if (itl != NonCorrectedSE3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->_id];

                g2o::SE3Quat Sli = Slw * Swi;
                g2o::EdgeSE3Expmap* el = new g2o::EdgeSE3Expmap();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->_id)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->_node->getCovisiblesByWeight(minFeat);
        for (KeyFrame* pKFn : vpConnectedKFs) {
            if (pKFn && pKFn != pParentKF && !pKF->_node->hasChild(pKFn) && !sLoopEdges.count(pKFn)) {
                if (!pKFn->isBad() && pKFn->_id < pKF->_id) {
                    if (sInsertedEdges.count(make_pair(min(pKF->_id, pKFn->_id), max(pKF->_id, pKFn->_id))))
                        continue;

                    g2o::SE3Quat Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSE3.find(pKFn);

                    if (itn != NonCorrectedSE3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->_id];

                    g2o::SE3Quat Sni = Snw * Swi;

                    g2o::EdgeSE3Expmap* en = new g2o::EdgeSE3Expmap();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->_id)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->_mutexMapUpdate);

    // SE3 Pose Recovering
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->_id;

        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(nIDi));
        g2o::SE3Quat CorrectedSiw = vSE3->estimate();
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        Quat eigR = CorrectedSiw.rotation();
        Vec3 eigt = CorrectedSiw.translation();

        SE3 Tiw(eigR, eigt);

        pKFi->setPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for (LandmarkPtr pMP : vpMPs) {
        if (pMP->isBad())
            continue;

        int nIDr;
        if (pMP->_correctedByKF == pCurKF->_id) {
            nIDr = pMP->_correctedReference;
        } else {
            KeyFrame* pRefKF = pMP->getReferenceKeyFrame();
            nIDr = pRefKF->_id;
        }

        g2o::SE3Quat Srw = vScw[nIDr];
        g2o::SE3Quat correctedSwr = vCorrectedSwc[nIDr];

        Vec3 P3Dw = pMP->getWorldPos();
        Vec3 eigCorrectedP3Dw = correctedSwr.map(Srw.map(P3Dw));

        pMP->setWorldPos(eigCorrectedP3Dw);
        pMP->updateNormalAndDepth();
    }
}
