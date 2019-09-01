#ifndef GRAPHNODE_H
#define GRAPHNODE_H

#include "System/Common.h"
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

class GraphNode {
public:
    SMART_POINTER_TYPEDEFS(GraphNode);

    GraphNode(KeyFrame* pKF, const bool spanningParentIsNotSet = true);

    ~GraphNode() = default;

    KeyFrame* getRefKF();

    // Covisibility graph functions
    void addConnection(KeyFrame* pKF, const unsigned int weight);
    void eraseConnection(KeyFrame* pKF);
    void eraseAllConnections();
    void updateConnections();
    void updateCovisibilityOrders();
    std::set<KeyFrame*> getConnectedKFs() const;
    std::vector<KeyFrame*> getCovisibles() const;
    std::vector<KeyFrame*> getBestNCovisibles(const unsigned int n) const;
    std::vector<KeyFrame*> getCovisiblesByWeight(const unsigned int w) const;
    unsigned int getWeight(KeyFrame* pKF);
    static bool weightComp(unsigned int a, unsigned int b) { return a > b; }

    // Spanning tree functions
    void setParent(KeyFrame* pParent);
    KeyFrame* getParent() const;
    void changeParent(KeyFrame* pNewParent);
    void addChild(KeyFrame* pChild);
    void eraseChild(KeyFrame* pChild);
    void recoverSpanningConnections();
    std::set<KeyFrame*> getChildrens() const;
    bool hasChild(KeyFrame* pKF) const;

    // Loop edges
    void addLoopEdge(KeyFrame* pLoopKF);
    std::set<KeyFrame*> getLoopEdges() const;
    bool hasLoopEdge() const;

protected:
    // KeyFrame of this node
    KeyFrame* const _ownerKF;

    // All connected KFs and ther weights
    std::map<KeyFrame*, unsigned int> _connectedKFs;

    // Minimum threshold for covisibility graph connection
    static constexpr unsigned int mWeightTh = 15;

    // Covisibility KFs in descending order on weights
    std::vector<KeyFrame*> _orderedKFs;
    std::vector<unsigned int> _orderedWeights;

    // Spanning tree
    KeyFrame* _parent = nullptr;
    std::set<KeyFrame*> _childrens;
    bool _spanningTreeIsNotSet;

    std::set<KeyFrame*> _loopEdges;

    mutable std::mutex _mutex;
};

#endif // GRAPHNODE_H
