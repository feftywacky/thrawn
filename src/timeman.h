#ifndef TIMEMAN_H
#define TIMEMAN_H

#include <cstdint>

// Two-bound time management. `soft` gates whether another iterative deepening
// iteration is started, `hard` aborts the search wherever it is. Both are
// durations in ms from `start`.
struct TimeManager {
    std::int64_t start = 0;
    std::int64_t soft  = 0;
    std::int64_t hard  = 0;
    // Bounds the depth-1 exemption in communicate(), the only thing that may
    // run past `hard`.
    std::int64_t panic = 0;

    // A clock is in play. False for `go depth` / `go infinite`.
    bool timeset = false;
    // The soft bound applies. False additionally for `go movetime`, which is an
    // instruction to use the whole slice.
    bool useSoft = false;

    void clear();

    // `timeLeft` and `movetimeMs` are -1 when the UCI token was absent;
    // `movestogo` is 0 for sudden death / Fischer.
    void init(int timeLeft, int inc, int movestogo, int movetimeMs,
              int rootMoveCount);

    std::int64_t elapsed() const;
    bool hard_expired() const;
    bool panic_expired() const;
    bool soft_expired(double scale) const;
};

extern TimeManager timeMan;

// UCI "Move Overhead", in ms.
extern int move_overhead;

// Multiplier on the soft bound: best-move stability x score trend x node effort.
// Both score diffs are (older - current), so positive means the score is falling.
double tm_soft_scale(int stability, int scoreDiffRecent, int scoreDiffPrev,
                     bool prevScoreKnown, long long bestMoveNodes,
                     long long totalNodes, bool decisiveScore);

#endif // TIMEMAN_H
