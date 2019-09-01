#include "GraphNode.h"
#include "KeyFrame.h"
#include "Landmark.h"
#include <list>

using namespace std;

GraphNode::GraphNode(KeyFrame* pKF, const bool spanningParentIsNotSet)
    : _ownerKF(pKF)
    , _spanningTreeIsNotSet(spanningParentIsNotSet)
{
}

KeyFrame* GraphNode::getRefKF() { return _ownerKF; }

void GraphNode::addConnection(KeyFrame* pKF, const unsigned int weight)
{
    bool update = false;

    {
        lock_guard<mutex> lock(_mutex);
        if (!_connectedKFs.count(pKF)) {
            _connectedKFs[pKF] = weight;
            update = true;
        } else if (_connectedKFs[pKF] != weight) {
            _connectedKFs[pKF] = weight;
            update = true;
        }
    }

    if (update)
        updateCovisibilityOrders();
}

void GraphNode::eraseConnection(KeyFrame* pKF)
{
    bool update = false;

    {
        lock_guard<mutex> lock(_mutex);
        if (_connectedKFs.count(pKF)) {
            _connectedKFs.erase(pKF);
            update = true;
        }
    }

    if (update)
        updateCovisibilityOrders();
}

void GraphNode::eraseAllConnections()
{
    for (auto& [pKF, w] : _connectedKFs)
        pKF->_node->eraseConnection(_ownerKF);

    _connectedKFs.clear();
    _orderedKFs.clear();
    _orderedWeights.clear();
}

void GraphNode::updateConnections()
{
    map<KeyFrame*, unsigned int> KFcounter;
    vector<LandmarkPtr> vpMPs = _ownerKF->getLandmarkMatches();

    for (LandmarkPtr pMP : vpMPs) {
        if (!pMP)
            continue;
        if (pMP->isBad())
            continue;

        map<KeyFrame*, size_t> obs = pMP->getObservations();

        for (auto& [pKF, idx] : obs) {
            if (pKF->_id == _ownerKF->_id)
                continue;
            KFcounter[pKF]++;
        }
    }

    if (KFcounter.empty())
        return;

    unsigned int maxWeight = 0;
    KeyFrame* pKFmax = nullptr;

    vector<pair<unsigned int, KeyFrame*>> vPairs;
    vPairs.reserve(KFcounter.size());
    for (auto& [pKF, w] : KFcounter) {
        if (w > maxWeight) {
            maxWeight = w;
            pKFmax = pKF;
        }
        if (w >= mWeightTh) {
            vPairs.push_back({ w, pKF });
            pKF->_node->addConnection(_ownerKF, w);
        }
    }

    if (vPairs.empty()) {
        vPairs.push_back({ maxWeight, pKFmax });
        pKFmax->_node->addConnection(_ownerKF, maxWeight);
    }

    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame*> lKFs;
    list<unsigned int> lWs;
    for (auto& [w, pKF] : vPairs) {
        lKFs.push_front(pKF);
        lWs.push_front(w);
    }

    {
        lock_guard<mutex> lock(_mutex);

        _connectedKFs = KFcounter;
        _orderedKFs = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        _orderedWeights = vector<unsigned int>(lWs.begin(), lWs.end());

        if (_spanningTreeIsNotSet && _ownerKF->_id != 0) {
            _parent = _orderedKFs.front();
            _parent->_node->addChild(_ownerKF);
            _spanningTreeIsNotSet = false;
        }
    }
}

void GraphNode::updateCovisibilityOrders()
{
    lock_guard<mutex> lock(_mutex);

    vector<pair<unsigned int, KeyFrame*>> vPairs;
    vPairs.reserve(_connectedKFs.size());
    for (auto& [pKF, w] : _connectedKFs)
        vPairs.push_back({ w, pKF });

    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame*> lKFs;
    list<unsigned int> lWs;
    for (auto& [w, pKF] : vPairs) {
        lKFs.push_front(pKF);
        lWs.push_front(w);
    }

    _orderedKFs = vector<KeyFrame*>(lKFs.begin(), lKFs.end());
    _orderedWeights = vector<unsigned int>(lWs.begin(), lWs.end());
}

set<KeyFrame*> GraphNode::getConnectedKFs() const
{
    lock_guard<mutex> lock(_mutex);
    set<KeyFrame*> s;
    for (auto& [pKF, w] : _connectedKFs)
        s.insert(pKF);
    return s;
}

vector<KeyFrame*> GraphNode::getCovisibles() const
{
    lock_guard<mutex> lock(_mutex);
    return _orderedKFs;
}

vector<KeyFrame*> GraphNode::getBestNCovisibles(const unsigned int n) const
{
    lock_guard<mutex> lock(_mutex);
    if (_orderedKFs.size() < n)
        return _orderedKFs;
    else
        return vector<KeyFrame*>(_orderedKFs.begin(), _orderedKFs.begin() + n);
}

vector<KeyFrame*> GraphNode::getCovisiblesByWeight(const unsigned int w) const
{
    lock_guard<mutex> lock(_mutex);

    if (_orderedKFs.empty())
        return vector<KeyFrame*>();

    auto it = upper_bound(_orderedWeights.begin(), _orderedWeights.end(), w, GraphNode::weightComp);
    if (it == _orderedWeights.end())
        return vector<KeyFrame*>();
    else {
        int n = it - _orderedWeights.begin();
        return vector<KeyFrame*>(_orderedKFs.begin(), _orderedKFs.begin() + n);
    }
}

unsigned int GraphNode::getWeight(KeyFrame* pKF)
{
    lock_guard<mutex> lock(_mutex);
    if (_connectedKFs.count(pKF))
        return _connectedKFs[pKF];
    else
        return 0;
}

void GraphNode::setParent(KeyFrame* pParent)
{
    lock_guard<mutex> lock(_mutex);
    _parent = pParent;
}

KeyFrame* GraphNode::getParent() const
{
    lock_guard<mutex> lock(_mutex);
    return _parent;
}

void GraphNode::changeParent(KeyFrame* pNewParent)
{
    lock_guard<mutex> lock(_mutex);
    _parent = pNewParent;
    pNewParent->_node->addChild(_ownerKF);
}

void GraphNode::addChild(KeyFrame* pChild)
{
    lock_guard<mutex> lock(_mutex);
    _childrens.insert(pChild);
}

void GraphNode::eraseChild(KeyFrame* pChild)
{
    lock_guard<mutex> lock(_mutex);
    _childrens.erase(pChild);
}

void GraphNode::recoverSpanningConnections()
{
    lock_guard<mutex> lock(_mutex);

    set<KeyFrame*> sParentCandidates;
    sParentCandidates.insert(_parent);

    while (!_childrens.empty()) {
        bool bContinue = false;

        unsigned int max = 0;
        KeyFrame* pC = nullptr;
        KeyFrame* pP = nullptr;

        for (KeyFrame* pKF : _childrens) {
            if (pKF->isBad())
                continue;

            vector<KeyFrame*> vpConnected = pKF->_node->getCovisibles();
            for (KeyFrame* pCovKF : vpConnected) {
                for (KeyFrame* pParentCandidateKF : sParentCandidates) {
                    if (pCovKF->_id == pParentCandidateKF->_id) {
                        unsigned int w = pKF->_node->getWeight(pCovKF);
                        if (w > max) {
                            pC = pKF;
                            pP = pCovKF;
                            max = w;
                            bContinue = true;
                        }
                    }
                }
            }
        }

        if (bContinue) {
            pC->_node->changeParent(pP);
            sParentCandidates.insert(pC);
            _childrens.erase(pC);
        } else
            break;
    }

    if (!_childrens.empty()) {
        for (KeyFrame* pChildKF : _childrens)
            pChildKF->_node->changeParent(_parent);
    }

    _parent->_node->eraseChild(_ownerKF);
}

set<KeyFrame*> GraphNode::getChildrens() const
{
    lock_guard<mutex> lock(_mutex);
    return _childrens;
}

bool GraphNode::hasChild(KeyFrame* pKF) const
{
    lock_guard<mutex> lock(_mutex);
    return _childrens.count(pKF);
}

void GraphNode::addLoopEdge(KeyFrame* pLoopKF)
{
    lock_guard<mutex> lock(_mutex);
    _ownerKF->setNotErase();
    _loopEdges.insert(pLoopKF);
}

set<KeyFrame*> GraphNode::getLoopEdges() const
{
    lock_guard<mutex> lock(_mutex);
    return _loopEdges;
}

bool GraphNode::hasLoopEdge() const
{
    lock_guard<mutex> lock(_mutex);
    return !_loopEdges.empty();
}
