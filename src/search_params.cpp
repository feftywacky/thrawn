#include "search_params.h"

#include <algorithm>
#include <cctype>

SearchParams searchParams;

namespace {

std::string normalize_name(const std::string& name) {
    std::string normalized;
    normalized.reserve(name.size());

    for (char ch : name) {
        normalized.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch))));
    }

    return normalized;
}

} // namespace

const std::array<SearchParameterMeta, SearchParameterCount>& search_parameter_metas() {
    static const std::array<SearchParameterMeta, SearchParameterCount> metas = {{
        {"SearchAspirationWindowDepth", 4, 1, 32, &searchParams.aspirationWindowDepth},
        {"SearchAspirationWindowSize", 24, 1, 500, &searchParams.aspirationWindowSize},
        {"SearchAspirationThreadDelta", 2, 0, 32, &searchParams.aspirationThreadDelta},
        {"SearchAspirationThreadCycle", 4, 1, 16, &searchParams.aspirationThreadCycle},
        {"SearchCheckExtension", 1, 0, 2, &searchParams.checkExtension},

        {"SearchHistoryMax", 16384, 1024, 65536, &searchParams.historyMax},
        {"SearchHistoryScoreCap", 6000, 512, 32768, &searchParams.historyScoreCap},
        {"SearchHistoryBonusDepthSquared", 16, 0, 64, &searchParams.historyBonusDepthSquared},
        {"SearchHistoryBonusDepthLinear", 64, 0, 256, &searchParams.historyBonusDepthLinear},

        {"SearchCounterMoveScore", 7000, 0, 30000, &searchParams.counterMoveScore},
        {"SearchCounterMoveHistoryDivisor", 64, 1, 512, &searchParams.counterMoveHistoryDivisor},
        {"SearchCounterMoveHistoryCap", 250, 0, 5000, &searchParams.counterMoveHistoryCap},

        {"SearchTtMoveScore", 30000, 0, 60000, &searchParams.ttMoveScore},
        {"SearchPvMoveScore", 20000, 0, 60000, &searchParams.pvMoveScore},
        {"SearchQueenPromotionScore", 10499, 0, 60000, &searchParams.queenPromotionScore},
        {"SearchKillerMoveScore1", 9000, 0, 60000, &searchParams.killerMoveScore1},
        {"SearchKillerMoveScore2", 8000, 0, 60000, &searchParams.killerMoveScore2},

        {"SearchReverseFutilityMaxDepth", 2, 0, 8, &searchParams.reverseFutilityMaxDepth},
        {"SearchReverseFutilityMargin1", 160, 0, 1000, &searchParams.reverseFutilityMargin1},
        {"SearchReverseFutilityMargin2", 300, 0, 1500, &searchParams.reverseFutilityMargin2},
        {"SearchReverseFutilityDepthFactor", 110, 0, 500, &searchParams.reverseFutilityDepthFactor},

        {"SearchRazorMaxDepth", 2, 0, 8, &searchParams.razorMaxDepth},
        {"SearchRazorMargin1", 250, 0, 1500, &searchParams.razorMargin1},
        {"SearchRazorMargin2", 450, 0, 2000, &searchParams.razorMargin2},
        {"SearchRazorMarginDepthN", 600, 0, 3000, &searchParams.razorMarginDepthN},

        {"SearchNullMoveMinDepth", 4, 1, 12, &searchParams.nullMoveMinDepth},
        {"SearchNullMoveBaseReduction", 2, 1, 6, &searchParams.nullMoveBaseReduction},
        {"SearchNullMoveDepthDivisor", 6, 1, 32, &searchParams.nullMoveDepthDivisor},
        {"SearchNullMoveEvalDivisor", 400, 1, 2000, &searchParams.nullMoveEvalDivisor},
        {"SearchNullMoveEvalBonusMax", 1, 0, 4, &searchParams.nullMoveEvalBonusMax},
        {"SearchNullMoveVerificationDepth", 8, 1, 16, &searchParams.nullMoveVerificationDepth},

        {"SearchFutilityMaxDepth", 3, 0, 8, &searchParams.futilityMaxDepth},
        {"SearchFutilityMargin1", 120, 0, 1000, &searchParams.futilityMargin1},
        {"SearchFutilityMargin2", 220, 0, 1500, &searchParams.futilityMargin2},
        {"SearchFutilityMargin3", 360, 0, 2000, &searchParams.futilityMargin3},

        {"SearchLateMovePruningMaxDepth", 3, 0, 8, &searchParams.lateMovePruningMaxDepth},
        {"SearchLateMovePruningDepth1", 8, 0, 64, &searchParams.lateMovePruningDepth1},
        {"SearchLateMovePruningDepth2", 12, 0, 128, &searchParams.lateMovePruningDepth2},
        {"SearchLateMovePruningDepth3", 24, 0, 256, &searchParams.lateMovePruningDepth3},

        {"SearchLmrFullDepthMoves", 3, 1, 16, &searchParams.lmrFullDepthMoves},
        {"SearchLmrReductionDepthLimit", 3, 1, 16, &searchParams.lmrReductionDepthLimit},
        {"SearchLmrBaseReduction", 0, 0, 6, &searchParams.lmrBaseReduction},
        {"SearchLmrNonPvDepth", 5, 1, 16, &searchParams.lmrNonPvDepth},
        {"SearchLmrMoveDepth1", 6, 1, 16, &searchParams.lmrMoveDepth1},
        {"SearchLmrMoveNumber1", 8, 1, 64, &searchParams.lmrMoveNumber1},
        {"SearchLmrMoveDepth2", 8, 1, 24, &searchParams.lmrMoveDepth2},
        {"SearchLmrMoveNumber2", 16, 1, 128, &searchParams.lmrMoveNumber2},
        {"SearchLmrGoodHistoryDivisor", 4, 1, 32, &searchParams.lmrGoodHistoryDivisor},
        {"SearchLmrBadHistoryDivisor", 4, 1, 32, &searchParams.lmrBadHistoryDivisor},

        {"SearchQsearchDeltaMargin", 200, 0, 1500, &searchParams.qsearchDeltaMargin},

        {"SearchSmpVoteScoreOffset", 14, 0, 200, &searchParams.smpVoteScoreOffset},
    }};

    return metas;
}

const SearchParameterMeta* find_search_parameter(const std::string& name) {
    const std::string normalizedName = normalize_name(name);

    for (const SearchParameterMeta& meta : search_parameter_metas()) {
        if (normalize_name(meta.name) == normalizedName) {
            return &meta;
        }
    }

    return nullptr;
}

bool set_search_parameter(const std::string& name, int value, int* appliedValue) {
    const SearchParameterMeta* meta = find_search_parameter(name);
    if (meta == nullptr) {
        return false;
    }

    const int clampedValue = std::clamp(value, meta->minValue, meta->maxValue);
    *meta->value = clampedValue;

    if (appliedValue != nullptr) {
        *appliedValue = clampedValue;
    }

    return true;
}

void reset_search_parameters() {
    for (const SearchParameterMeta& meta : search_parameter_metas()) {
        *meta.value = meta.defaultValue;
    }
}
