#include "timeman.h"

#include <algorithm>

#include "constants.h"
#include "uci.h" // for get_time_ms()

TimeManager timeMan;

int move_overhead = TM_MOVE_OVERHEAD_DEFAULT;

void TimeManager::clear()
{
    start   = 0;
    soft    = 0;
    hard    = 0;
    timeset = false;
    useSoft = false;
}

void TimeManager::init(int timeLeft, int inc, int movestogo, int movetimeMs,
                       int rootMoveCount)
{
    clear();
    start = get_time_ms();

    // `go movetime N` is an instruction, not a budget: spend the whole slice.
    if (movetimeMs >= 0)
    {
        timeset = true;
        useSoft = false;
        hard    = std::max<std::int64_t>(1, movetimeMs - move_overhead);
        soft    = hard;
        return;
    }

    // No clock at all: `go depth`, `go infinite`, or a bare `go`.
    if (timeLeft < 0)
        return;

    timeset = true;
    useSoft = true;

    const double overhead = static_cast<double>(move_overhead);
    const double time     = static_cast<double>(std::max(0, timeLeft));

    double softMs = 0.0;
    double hardMs = 0.0;

    if (movestogo <= 0)
    {
        // Berserk lets the horizon term go negative, which collapses the
        // allocation once remaining time drops below horizon * overhead.
        // Flooring it at zero leaves a low clock spending a fixed share of what
        // is left, and changes nothing while the increment covers the overhead.
        const double horizon = TM_FISCHER_HORIZON * (static_cast<double>(inc) - overhead);
        const double total   = time + std::max(0.0, horizon);

        softMs = std::min(time * TM_SD_SOFT_CAP, total * TM_SD_SOFT_SCALE);
        hardMs = std::min(time * TM_SD_HARD_CAP - overhead, softMs * TM_SD_HARD_MULT)
               - TM_SAFETY_MS;
    }
    else
    {
        const double total = std::max(
            1.0, time + static_cast<double>(movestogo) * static_cast<double>(inc) - overhead);
        const double horizon =
            std::max(1.0, static_cast<double>(movestogo) / TM_CYC_HORIZON_DIV);

        softMs = std::min(time * TM_CYC_SOFT_CAP, TM_CYC_SOFT_SCALE * total / horizon);
        hardMs = std::min(time * TM_CYC_HARD_CAP - overhead, softMs * TM_CYC_HARD_MULT)
               - TM_SAFETY_MS;
    }

    // No formula above may put a bound past the flag.
    const std::int64_t panic = std::max<std::int64_t>(1, timeLeft - move_overhead);
    hard = std::clamp<std::int64_t>(static_cast<std::int64_t>(hardMs), 1, panic);
    soft = std::clamp<std::int64_t>(static_cast<std::int64_t>(softMs), 1, hard);

    // One legal move: nothing to decide, and no ponder move to look for.
    if (rootMoveCount == 1)
    {
        hard = std::min<std::int64_t>(hard, TM_SINGLE_MOVE_MS);
        soft = std::min(soft, hard);
    }
}

std::int64_t TimeManager::elapsed() const
{
    return get_time_ms() - start;
}

bool TimeManager::hard_expired() const
{
    return timeset && elapsed() >= hard;
}

bool TimeManager::soft_expired(double scale) const
{
    if (!timeset || !useSoft)
        return false;

    return static_cast<double>(elapsed()) > static_cast<double>(soft) * scale;
}

double tm_soft_scale(int stability, int scoreDiffRecent, int scoreDiffPrev,
                     bool prevScoreKnown, long long bestMoveNodes,
                     long long totalNodes, bool decisiveScore)
{
    int recent = scoreDiffRecent;
    int prev   = scoreDiffPrev;

    // First search of a game: the within-search trend carries the whole term.
    if (!prevScoreKnown)
    {
        recent *= 2;
        prev = 0;
    }

    // (a) An unchanged root move across iterations means a settled position.
    const double stabilityFactor =
        TM_STABILITY_BASE -
        TM_STABILITY_STEP * std::clamp(stability, 0, TM_STABILITY_MAX);

    // (b) Only a falling score buys time; stability already covers the rest.
    double scoreFactor = TM_SCORE_BASE +
                         TM_SCORE_RECENT_COEF * std::max(0, recent) +
                         TM_SCORE_PREV_COEF * std::max(0, prev);
    scoreFactor = std::clamp(scoreFactor, TM_SCORE_MIN, TM_SCORE_MAX);

    // (c) A tree spread across many root moves is an unresolved position.
    double nodeFactor;
    if (decisiveScore)
    {
        nodeFactor = TM_NODE_DECISIVE;
    }
    else
    {
        const double pctNotBest =
            totalNodes > 0
                ? 1.0 - static_cast<double>(bestMoveNodes) / static_cast<double>(totalNodes)
                : 0.0;
        nodeFactor = std::max(TM_NODE_MIN, pctNotBest * TM_NODE_COEF + TM_NODE_BASE);
    }

    // Peaks at ~5.90, just under TM_SD_HARD_MULT (5.928), so the soft path can
    // reach the hard bound but not overshoot it. Preserve that if retuning.
    return stabilityFactor * scoreFactor * nodeFactor;
}
