#ifndef SEARCH_PARAMS_H
#define SEARCH_PARAMS_H

#include <array>
#include <cstddef>
#include <string>

struct SearchParams {
    int aspirationWindowDepth = 4;
    int aspirationWindowSize = 24;
    int aspirationThreadDelta = 2;
    int aspirationThreadCycle = 4;
    int checkExtension = 1;

    int historyMax = 16384;
    int historyScoreCap = 6000;
    int historyBonusDepthSquared = 16;
    int historyBonusDepthLinear = 64;

    int counterMoveScore = 7000;
    int counterMoveHistoryDivisor = 64;
    int counterMoveHistoryCap = 250;

    int ttMoveScore = 30000;
    int pvMoveScore = 20000;
    int queenPromotionScore = 10499;
    int killerMoveScore1 = 9000;
    int killerMoveScore2 = 8000;

    int reverseFutilityMaxDepth = 2;
    int reverseFutilityMargin1 = 160;
    int reverseFutilityMargin2 = 300;
    int reverseFutilityDepthFactor = 110;

    int razorMaxDepth = 2;
    int razorMargin1 = 250;
    int razorMargin2 = 450;
    int razorMarginDepthN = 600;

    int nullMoveMinDepth = 4;
    int nullMoveBaseReduction = 2;
    int nullMoveDepthDivisor = 6;
    int nullMoveEvalDivisor = 400;
    int nullMoveEvalBonusMax = 1;
    int nullMoveVerificationDepth = 8;

    int futilityMaxDepth = 3;
    int futilityMargin1 = 120;
    int futilityMargin2 = 220;
    int futilityMargin3 = 360;

    int lateMovePruningMaxDepth = 3;
    int lateMovePruningDepth1 = 8;
    int lateMovePruningDepth2 = 12;
    int lateMovePruningDepth3 = 24;

    int lmrFullDepthMoves = 3;
    int lmrReductionDepthLimit = 3;
    int lmrBaseReduction = 0;
    int lmrNonPvDepth = 5;
    int lmrMoveDepth1 = 6;
    int lmrMoveNumber1 = 8;
    int lmrMoveDepth2 = 8;
    int lmrMoveNumber2 = 16;
    int lmrGoodHistoryDivisor = 4;
    int lmrBadHistoryDivisor = 4;

    int qsearchDeltaMargin = 200;

    int smpVoteScoreOffset = 14;
};

struct SearchParameterMeta {
    const char* name;
    int defaultValue;
    int minValue;
    int maxValue;
    int* value;
};

extern SearchParams searchParams;

constexpr std::size_t SearchParameterCount = 51;

const std::array<SearchParameterMeta, SearchParameterCount>& search_parameter_metas();
const SearchParameterMeta* find_search_parameter(const std::string& name);
bool set_search_parameter(const std::string& name, int value, int* appliedValue = nullptr);
void reset_search_parameters();

#endif // SEARCH_PARAMS_H
