#include "KeyFrameDatabase.h"
#include "Core/Frame.h"
#include "Core/GraphNode.h"
#include "Core/KeyFrame.h"
#include <DBoW3/BowVector.h>
#include <mutex>

using namespace std;

KeyFrameDatabase::KeyFrameDatabase(const VocabularyPtr voc)
    : _vocabulary(voc)
{
    _invertedFile.resize(voc->size());
}

void KeyFrameDatabase::add(KeyFrame* pKF)
{
    unique_lock<mutex> lock(_mutex);

    for (const auto& [wordId, wordValue] : pKF->_bowVec)
        _invertedFile[wordId].push_back(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(_mutex);

    // Erase elements in the Inverse File for the entry
    for (const auto& [wordId, wordValue] : pKF->_bowVec) {
        // List of keyframes that share the word
        list<KeyFrame*>& lKFs = _invertedFile[wordId];

        for (list<KeyFrame*>::iterator lit = lKFs.begin(); lit != lKFs.end(); lit++) {
            if (pKF == *lit) {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    _invertedFile.clear();
    _invertedFile.resize(_vocabulary->size());
}

vector<KeyFrame*> KeyFrameDatabase::detectLoopCandidates(KeyFrame* pKF, float minScore)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->_node->getConnectedKFs();
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(_mutex);

        for (const auto& [wordId, wordValue] : pKF->_bowVec) {
            list<KeyFrame*>& lKFs = _invertedFile[wordId];

            for (KeyFrame* pKFi : lKFs) {
                if (pKFi->_loopQuery != pKF->_id) {
                    pKFi->_loopWords = 0;
                    if (!spConnectedKeyFrames.count(pKFi)) {
                        pKFi->_loopQuery = pKF->_id;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->_loopWords++;
            }
        }
    }

    if (lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float, KeyFrame*>> lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (KeyFrame* pKFi : lKFsSharingWords) {
        if (pKFi->_loopWords > maxCommonWords)
            maxCommonWords = pKFi->_loopWords;
    }

    int minCommonWords = maxCommonWords * 0.8f;
    int nscores = 0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for (KeyFrame* pKFi : lKFsSharingWords) {
        if (pKFi->_loopWords > minCommonWords) {
            nscores++;

            float si = static_cast<float>(_vocabulary->score(pKF->_bowVec, pKFi->_bowVec));

            pKFi->_loopScore = si;
            if (si >= minScore)
                lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float, KeyFrame*>> lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for (auto& [score, pKFi] : lScoreAndMatch) {
        vector<KeyFrame*> vpNeighs = pKFi->_node->getBestNCovisibles(10);

        float bestScore = score;
        float accScore = score;
        KeyFrame* pBestKF = pKFi;
        for (KeyFrame* pKF2 : vpNeighs) {
            if (pKF2->_loopQuery == pKF->_id && pKF2->_loopWords > minCommonWords) {
                accScore += pKF2->_loopScore;
                if (pKF2->_loopScore > bestScore) {
                    pBestKF = pKF2;
                    bestScore = pKF2->_loopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f * bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for (auto& [score, pKFi] : lAccScoreAndMatch) {
        if (score > minScoreToRetain) {
            if (!spAlreadyAddedKF.count(pKFi)) {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}
