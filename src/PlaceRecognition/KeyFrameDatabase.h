#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include "System/Common.h"
#include <DBoW3/Vocabulary.h>
#include <list>
#include <mutex>
#include <set>
#include <vector>

class KeyFrameDatabase {
public:
    KeyFrameDatabase(const VocabularyPtr voc);

    void add(KeyFrame* pKF);

    void erase(KeyFrame* pKF);

    void clear();

    // Loop Detection
    std::vector<KeyFrame*> detectLoopCandidates(KeyFrame* pKF, float minScore);

protected:
    // Associated vocabulary
    const VocabularyPtr _vocabulary;

    // Inverted file
    std::vector<std::list<KeyFrame*>> _invertedFile;

    // Mutex
    std::mutex _mutex;
};

#endif
