#include "nnue.h"

#include "bitboard.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

#if defined(USE_NEON)
#include <arm_neon.h>
#endif

namespace {

constexpr char kExpectedMagic[8] = {'T', 'H', 'N', 'N', 'U', 'E', '\0', '\1'};
constexpr uint32_t kCurrentVersion = 7;
constexpr uint32_t kExpectedNumFeatures = thrawn::NNUE_INPUT_FEATURES;
constexpr uint32_t kExpectedFtSize = thrawn::NNUE_ACCUMULATOR_SIZE;
constexpr uint32_t kExpectedL1Size = thrawn::NNUE_L1_SIZE;
constexpr uint32_t kExpectedL2Size = thrawn::NNUE_L2_SIZE;
constexpr uint32_t kExpectedOutputPerspective = 1;
constexpr const char* kExpectedFeatureSet = "halfkp_v1";

constexpr int kHalfkpBuckets = 10;
constexpr int kHalfkpStridePerKing = kHalfkpBuckets * 64; // 640
constexpr int kL1InputSize = static_cast<int>(kExpectedFtSize) * 2; // [us_acc | them_acc]

constexpr int32_t kFtActivationMax = 127;
constexpr int32_t kDenseActivationMax = 64;
constexpr double kCpPerStockfishScore = 100.0 / 208.0;

// Thrawn's search constants are classical cp values: the HCE fallback piece
// values, futility margins, aspiration window, and UCI reporting are all cp-based.
constexpr bool kSearchUsesCentipawns = true;

struct HalfkpSelfTestCase {
    int piece;
    int piece_square;
    int king_square;
    bool white_perspective;
    int expected_index;
};

constexpr std::array<HalfkpSelfTestCase, 4> kHalfkpSelfTests{{
    {P, 48, 60, true, 2568},   // white pawn a2, white king e1, white perspective
    {R, 63, 4, false, 3071},   // white rook h1, black king e8, black perspective
    {n, 9, 62, true, 4081},    // black knight b7, white king g1, white perspective
    {b, 26, 1, false, 922},    // black bishop c5, black king b8, black perspective
}};

struct LoadedNetwork {
    uint32_t version = 0;
    uint32_t output_perspective = 0;

    float ft_scale = 0.0f;
    float l1_scale = 0.0f;
    float l2_scale = 0.0f;
    float out_scale = 0.0f;
    float score_scale = 0.0f;

    int32_t qa = 0;
    int32_t q1 = 0;
    int32_t q2 = 0;

    std::array<int16_t, kExpectedFtSize> ft_bias{};
    std::vector<int16_t> ft_weight;

    std::array<int32_t, kExpectedL1Size> l1_bias{};
    alignas(64) std::array<std::array<int8_t, kL1InputSize>, kExpectedL1Size> l1_weight_t{}; // [out][in]

    std::array<int32_t, kExpectedL2Size> l2_bias{};
    alignas(64) std::array<std::array<int8_t, kExpectedL1Size>, kExpectedL2Size> l2_weight_t{}; // [out][in]

    int32_t out_bias = 0;
    alignas(64) std::array<int8_t, kExpectedL2Size> out_weight{};

    std::string description;
    std::string loaded_path;
};

std::mutex g_network_mutex;
std::shared_ptr<const LoadedNetwork> g_network_owner;
std::atomic<const LoadedNetwork*> g_network_raw{nullptr};

bool host_is_little_endian() {
    const uint16_t endian_probe = 0x0102;
    return *reinterpret_cast<const unsigned char*>(&endian_probe) == 0x02;
}

template <typename T>
bool read_scalar_le(std::ifstream& stream, T& value) {
    std::array<unsigned char, sizeof(T)> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        return false;
    }

    if (host_is_little_endian()) {
        std::memcpy(&value, bytes.data(), sizeof(T));
        return true;
    }

    std::array<unsigned char, sizeof(T)> reversed{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        reversed[i] = bytes[bytes.size() - 1 - i];
    }
    std::memcpy(&value, reversed.data(), sizeof(T));
    return true;
}

template <typename T>
bool read_vector_le(std::ifstream& stream, std::vector<T>& values, std::size_t count) {
    values.resize(count);
    if (count == 0) {
        return true;
    }

    if constexpr (sizeof(T) == 1) {
        stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count));
        return static_cast<bool>(stream);
    } else if (host_is_little_endian()) {
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(count * sizeof(T)));
        return static_cast<bool>(stream);
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            if (!read_scalar_le(stream, values[i])) {
                return false;
            }
        }
        return true;
    }
}

std::string trim_copy(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }

    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(start, end - start);
}

bool looks_like_absolute_path(const std::string& path) {
#ifdef _WIN32
    return path.size() > 2 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':';
#else
    return !path.empty() && path[0] == '/';
#endif
}

std::vector<std::string> candidate_paths(const std::string& requested_path) {
    std::string path = trim_copy(requested_path);
    if (path.empty()) {
        path = "model_v6.nnue";
    }

    std::vector<std::string> candidates{path};
    if (!looks_like_absolute_path(path) && path.rfind("./", 0) != 0 && path.rfind("../", 0) != 0) {
        candidates.push_back("src/" + path);
    }
    return candidates;
}

std::string read_feature_set(std::ifstream& stream) {
    char feature_set_raw[16] = {};
    stream.read(feature_set_raw, sizeof(feature_set_raw));
    if (!stream) {
        return {};
    }

    std::size_t length = 0;
    while (length < sizeof(feature_set_raw) && feature_set_raw[length] != '\0') {
        ++length;
    }
    return std::string(feature_set_raw, length);
}

bool parse_header_scale(float scale,
                        const char* scale_name,
                        int32_t max_allowed,
                        int32_t& out_cap,
                        std::string& error) {
    if (!std::isfinite(scale) || scale <= 0.0f) {
        error = std::string("invalid ") + scale_name;
        return false;
    }

    if (scale > static_cast<float>(max_allowed)) {
        error = std::string(scale_name) + " is too large";
        return false;
    }

    const long rounded = std::lround(scale);
    if (rounded <= 0 || rounded > max_allowed) {
        error = std::string("invalid ") + scale_name;
        return false;
    }

    out_cap = static_cast<int32_t>(rounded);
    return true;
}

bool parse_positive_bounded_float(float scale,
                                  const char* scale_name,
                                  float max_allowed,
                                  std::string& error) {
    if (!std::isfinite(scale) || scale <= 0.0f || scale > max_allowed) {
        error = std::string("invalid ") + scale_name;
        return false;
    }
    return true;
}

int64_t div_round_by_scale(int64_t value, int32_t scale) {
    if (scale <= 0) {
        return 0;
    }

    const int64_t divisor = scale;
    if (value >= 0) {
        return (value + divisor / 2) / divisor;
    }
    return -((-value + divisor / 2) / divisor);
}

int engine_to_model_square(int square) {
    return square ^ 56;
}

int flip_vertical(int square) {
    return ((7 - (square / 8)) * 8) + (square % 8);
}

int oriented_square(int engine_square, bool white_perspective) {
    const int white_square = engine_to_model_square(engine_square);
    return white_perspective ? white_square : flip_vertical(white_square);
}

bool is_white_piece(int piece) {
    return piece >= P && piece <= K;
}

bool is_king_piece(int piece) {
    return piece == K || piece == k;
}

int piece_type_index(int piece) {
    switch (piece) {
        case P:
        case p:
            return 0;
        case N:
        case n:
            return 1;
        case B:
        case b:
            return 2;
        case R:
        case r:
            return 3;
        case Q:
        case q:
            return 4;
        default:
            return -1;
    }
}

int relative_color_bit(int piece, bool white_perspective) {
    return white_perspective
        ? (is_white_piece(piece) ? 0 : 1)
        : (is_white_piece(piece) ? 1 : 0);
}

int halfkp_index(int piece,
                 int piece_square,
                 int our_king_square,
                 bool white_perspective) {
    const int ptype = piece_type_index(piece);
    if (ptype < 0 || piece_square < 0 || piece_square >= BOARD_SIZE ||
        our_king_square < 0 || our_king_square >= BOARD_SIZE) {
        return -1;
    }

    const int oriented_ksq = oriented_square(our_king_square, white_perspective);
    const int oriented_sq = oriented_square(piece_square, white_perspective);
    const int piece_bucket = ptype * 2 + relative_color_bit(piece, white_perspective);
    const int p_index = piece_bucket * 64 + oriented_sq;
    return oriented_ksq * kHalfkpStridePerKing + p_index;
}

const LoadedNetwork* current_network() {
    return g_network_raw.load(std::memory_order_acquire);
}

void clear_state(thrawn::NnueState& state) {
    state.white_acc.fill(0);
    state.black_acc.fill(0);
    state.piece_list.fill(0);
    state.square_list.fill(0);
    state.index_by_square.fill(-1);
    state.piece_count = 0;
    state.white_king_sq = -1;
    state.black_king_sq = -1;
    state.valid = false;
}

void copy_ft_bias(std::array<int16_t, kExpectedFtSize>& accumulator,
                  const std::array<int16_t, kExpectedFtSize>& bias) {
    accumulator = bias;
}

enum class AccumulatorUpdateOp { Add, Subtract };

#if defined(USE_AVX2)
static inline void acc_add_row_avx2(int16_t* acc, const int16_t* row) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
        const __m256i a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i + 16));
        const __m256i r2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i + 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_add_epi16(a, r));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i + 16), _mm256_add_epi16(a2, r2));
    }
}

static inline void acc_sub_row_avx2(int16_t* acc, const int16_t* row) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
        const __m256i a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i + 16));
        const __m256i r2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i + 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_sub_epi16(a, r));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i + 16), _mm256_sub_epi16(a2, r2));
    }
}
#endif

#if defined(USE_NEON)
static inline void acc_add_row_neon(int16_t* acc, const int16_t* row) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
        const int16x8_t a = vld1q_s16(acc + i);
        const int16x8_t r = vld1q_s16(row + i);
        const int16x8_t a2 = vld1q_s16(acc + i + 8);
        const int16x8_t r2 = vld1q_s16(row + i + 8);
        vst1q_s16(acc + i, vaddq_s16(a, r));
        vst1q_s16(acc + i + 8, vaddq_s16(a2, r2));
    }
}

static inline void acc_sub_row_neon(int16_t* acc, const int16_t* row) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
        const int16x8_t a = vld1q_s16(acc + i);
        const int16x8_t r = vld1q_s16(row + i);
        const int16x8_t a2 = vld1q_s16(acc + i + 8);
        const int16x8_t r2 = vld1q_s16(row + i + 8);
        vst1q_s16(acc + i, vsubq_s16(a, r));
        vst1q_s16(acc + i + 8, vsubq_s16(a2, r2));
    }
}
#endif

bool update_accumulator_row(std::array<int16_t, kExpectedFtSize>& accumulator,
                            const std::vector<int16_t>& weights,
                            int feature,
                            AccumulatorUpdateOp op) {
    if (feature < 0) {
        return false;
    }

    const std::size_t offset = static_cast<std::size_t>(feature) * kExpectedFtSize;
    if (offset + kExpectedFtSize > weights.size()) {
        return false;
    }

    const int16_t* row = weights.data() + offset;
    int16_t* acc = accumulator.data();

#if defined(USE_AVX2)
    if (op == AccumulatorUpdateOp::Add) {
        acc_add_row_avx2(acc, row);
    } else {
        acc_sub_row_avx2(acc, row);
    }
#elif defined(USE_NEON)
    if (op == AccumulatorUpdateOp::Add) {
        acc_add_row_neon(acc, row);
    } else {
        acc_sub_row_neon(acc, row);
    }
#else
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        const int32_t value = (op == AccumulatorUpdateOp::Add)
            ? static_cast<int32_t>(acc[i]) + row[i]
            : static_cast<int32_t>(acc[i]) - row[i];
        acc[i] = static_cast<int16_t>(value);
    }
#endif

    return true;
}

bool add_accumulator_row(std::array<int16_t, kExpectedFtSize>& accumulator,
                         const std::vector<int16_t>& weights,
                         int feature) {
    return update_accumulator_row(accumulator, weights, feature, AccumulatorUpdateOp::Add);
}

bool subtract_accumulator_row(std::array<int16_t, kExpectedFtSize>& accumulator,
                              const std::vector<int16_t>& weights,
                              int feature) {
    return update_accumulator_row(accumulator, weights, feature, AccumulatorUpdateOp::Subtract);
}

bool track_piece(thrawn::NnueState& state, int piece, int square) {
    if (square < 0 || square >= BOARD_SIZE || state.piece_count >= thrawn::NNUE_MAX_PIECES) {
        return false;
    }
    if (state.index_by_square[square] != -1) {
        return false;
    }

    if (piece == K && state.white_king_sq != -1) {
        return false;
    }
    if (piece == k && state.black_king_sq != -1) {
        return false;
    }

    const int index = state.piece_count;
    state.piece_list[index] = static_cast<uint8_t>(piece);
    state.square_list[index] = static_cast<uint8_t>(square);
    state.index_by_square[square] = static_cast<int8_t>(index);
    state.piece_count = static_cast<uint8_t>(state.piece_count + 1);

    if (piece == K) {
        state.white_king_sq = static_cast<int8_t>(square);
    } else if (piece == k) {
        state.black_king_sq = static_cast<int8_t>(square);
    }

    return true;
}

bool untrack_piece(thrawn::NnueState& state, int piece, int square) {
    if (square < 0 || square >= BOARD_SIZE) {
        return false;
    }

    const int index = state.index_by_square[square];
    if (index < 0 || index >= static_cast<int>(state.piece_count) || state.piece_list[index] != piece) {
        return false;
    }

    if (piece == K) {
        state.white_king_sq = -1;
    } else if (piece == k) {
        state.black_king_sq = -1;
    }

    const int last_index = static_cast<int>(state.piece_count) - 1;
    state.index_by_square[square] = -1;

    if (index != last_index) {
        const uint8_t moved_piece = state.piece_list[last_index];
        const uint8_t moved_square = state.square_list[last_index];
        state.piece_list[index] = moved_piece;
        state.square_list[index] = moved_square;
        state.index_by_square[moved_square] = static_cast<int8_t>(index);
    }

    state.piece_list[last_index] = 0;
    state.square_list[last_index] = 0;
    state.piece_count = static_cast<uint8_t>(state.piece_count - 1);
    return true;
}

bool refresh_perspective_accumulator(thrawn::NnueState& state,
                                     const LoadedNetwork& network,
                                     bool white_perspective) {
    const int our_king_square = white_perspective ? state.white_king_sq : state.black_king_sq;
    if (our_king_square < 0 || our_king_square >= BOARD_SIZE) {
        return false;
    }

    auto& acc = white_perspective ? state.white_acc : state.black_acc;
    copy_ft_bias(acc, network.ft_bias);

    for (int i = 0; i < static_cast<int>(state.piece_count); ++i) {
        const int piece = state.piece_list[i];
        const int square = state.square_list[i];
        if (is_king_piece(piece)) {
            continue;
        }

        const int feature = halfkp_index(piece, square, our_king_square, white_perspective);
        if (feature < 0 || !add_accumulator_row(acc, network.ft_weight, feature)) {
            return false;
        }
    }

    return true;
}

bool patch_piece_for_available_perspectives(thrawn::NnueState& state,
                                            const LoadedNetwork& network,
                                            int piece,
                                            int square,
                                            AccumulatorUpdateOp op) {
    if (is_king_piece(piece)) {
        return true;
    }

    bool patched_any = false;

    if (state.white_king_sq >= 0 && state.white_king_sq < BOARD_SIZE) {
        const int feature = halfkp_index(piece, square, state.white_king_sq, true);
        if (feature < 0) {
            return false;
        }
        const bool ok = (op == AccumulatorUpdateOp::Add)
            ? add_accumulator_row(state.white_acc, network.ft_weight, feature)
            : subtract_accumulator_row(state.white_acc, network.ft_weight, feature);
        if (!ok) {
            return false;
        }
        patched_any = true;
    }

    if (state.black_king_sq >= 0 && state.black_king_sq < BOARD_SIZE) {
        const int feature = halfkp_index(piece, square, state.black_king_sq, false);
        if (feature < 0) {
            return false;
        }
        const bool ok = (op == AccumulatorUpdateOp::Add)
            ? add_accumulator_row(state.black_acc, network.ft_weight, feature)
            : subtract_accumulator_row(state.black_acc, network.ft_weight, feature);
        if (!ok) {
            return false;
        }
        patched_any = true;
    }

    return patched_any;
}

bool refresh_perspective_accumulator_from_board(const thrawn::Position* pos,
                                                thrawn::NnueState& state,
                                                const LoadedNetwork& network,
                                                bool white_perspective) {
    if (pos == nullptr) {
        return false;
    }

    auto& acc = white_perspective ? state.white_acc : state.black_acc;
    copy_ft_bias(acc, network.ft_bias);

    const int our_king_square = white_perspective
        ? get_lsb_index(pos->piece_bitboards[K])
        : get_lsb_index(pos->piece_bitboards[k]);
    if (our_king_square < 0 || our_king_square >= BOARD_SIZE) {
        return false;
    }

    if (white_perspective) {
        state.white_king_sq = static_cast<int8_t>(our_king_square);
    } else {
        state.black_king_sq = static_cast<int8_t>(our_king_square);
    }

    for (int piece = P; piece <= k; ++piece) {
        if (is_king_piece(piece)) {
            continue;
        }

        uint64_t bitboard = pos->piece_bitboards[piece];
        while (bitboard) {
            const int square = pop_lsb(bitboard);
            const int feature = halfkp_index(piece, square, our_king_square, white_perspective);
            if (feature < 0 || !add_accumulator_row(acc, network.ft_weight, feature)) {
                return false;
            }
        }
    }

    return true;
}

bool add_piece_to_state(thrawn::NnueState& state,
                        const LoadedNetwork& network,
                        int piece,
                        int square) {
    if (!is_king_piece(piece)) {
        if (!patch_piece_for_available_perspectives(state, network, piece, square, AccumulatorUpdateOp::Add)) {
            return false;
        }
    }

    if (!track_piece(state, piece, square)) {
        return false;
    }

    return true;
}

bool remove_piece_from_state(thrawn::NnueState& state,
                             const LoadedNetwork& network,
                             int piece,
                             int square) {
    if (!is_king_piece(piece)) {
        if (!patch_piece_for_available_perspectives(state, network, piece, square, AccumulatorUpdateOp::Subtract)) {
            return false;
        }
    }

    return untrack_piece(state, piece, square);
}

bool build_state_from_board(const thrawn::Position* pos,
                            const LoadedNetwork& network,
                            thrawn::NnueState& out_state) {
    clear_state(out_state);

    for (int piece = P; piece <= k; ++piece) {
        uint64_t bitboard = pos->piece_bitboards[piece];
        while (bitboard) {
            const int square = pop_lsb(bitboard);
            if (!track_piece(out_state, piece, square)) {
                clear_state(out_state);
                return false;
            }
        }
    }

    if (out_state.white_king_sq < 0 || out_state.black_king_sq < 0) {
        clear_state(out_state);
        return false;
    }

    if (!refresh_perspective_accumulator(out_state, network, true)) {
        clear_state(out_state);
        return false;
    }

    if (!refresh_perspective_accumulator(out_state, network, false)) {
        clear_state(out_state);
        return false;
    }

    out_state.valid = true;
    return true;
}

const thrawn::NnueState* state_for_evaluation(const thrawn::Position* pos,
                                              const LoadedNetwork& network,
                                              thrawn::NnueState& refreshed_state) {
    if (pos->ply >= 0 && pos->ply <= MAX_DEPTH) {
        const thrawn::NnueState& live_state = pos->nnue_stack[pos->ply];
        if (live_state.valid) {
            return &live_state;
        }
    }

    if (!build_state_from_board(pos, network, refreshed_state)) {
        return nullptr;
    }
    return &refreshed_state;
}

#if defined(USE_AVX2)
void pack_clipped_u8_avx2(uint8_t* out,
                          const int16_t* us,
                          const int16_t* them,
                          int ft_scale_i) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i cap = _mm256_set1_epi16(static_cast<int16_t>(ft_scale_i));

    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(us + i));
        __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(us + i + 16));
        a0 = _mm256_min_epi16(_mm256_max_epi16(a0, zero), cap);
        a1 = _mm256_min_epi16(_mm256_max_epi16(a1, zero), cap);
        __m256i packed = _mm256_packus_epi16(a0, a1);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), packed);
    }

    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(them + i));
        __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(them + i + 16));
        a0 = _mm256_min_epi16(_mm256_max_epi16(a0, zero), cap);
        a1 = _mm256_min_epi16(_mm256_max_epi16(a1, zero), cap);
        __m256i packed = _mm256_packus_epi16(a0, a1);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + kExpectedFtSize + i), packed);
    }
}

int32_t dot_u8_i8_avx2(const uint8_t* x, const int8_t* w, int count) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 127 < count; i += 128) {
        const __m256i xv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        const __m256i wv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        const __m256i xv1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + 32));
        const __m256i wv1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + 32));
        const __m256i xv2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + 64));
        const __m256i wv2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + 64));
        const __m256i xv3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + 96));
        const __m256i wv3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + 96));

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(xv0, wv0), ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(xv1, wv1), ones));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(xv2, wv2), ones));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(xv3, wv3), ones));
    }

    acc0 = _mm256_add_epi32(acc0, acc1);
    acc2 = _mm256_add_epi32(acc2, acc3);
    acc0 = _mm256_add_epi32(acc0, acc2);

    for (; i < count; i += 32) {
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        const __m256i prod16 = _mm256_maddubs_epi16(xv, wv);
        const __m256i prod32 = _mm256_madd_epi16(prod16, ones);
        acc0 = _mm256_add_epi32(acc0, prod32);
    }

    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc0);

    int32_t sum = 0;
    for (int i = 0; i < 8; ++i) {
        sum += lanes[i];
    }
    return sum;
}
#endif

#if defined(USE_NEON)
void pack_clipped_u8_neon(uint8_t* out,
                          const int16_t* us,
                          const int16_t* them,
                          int ft_scale_i) {
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t cap = vdupq_n_s16(static_cast<int16_t>(ft_scale_i));

    auto pack_half = [&](const int16_t* src, uint8_t* dst) {
        for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
            const int16x8_t a0 = vminq_s16(vmaxq_s16(vld1q_s16(src + i), zero), cap);
            const int16x8_t a1 = vminq_s16(vmaxq_s16(vld1q_s16(src + i + 8), zero), cap);
            const uint8x16_t p = vcombine_u8(vqmovun_s16(a0), vqmovun_s16(a1));
            vst1q_u8(dst + i, p);
        }
    };

    pack_half(us, out);
    pack_half(them, out + kExpectedFtSize);
}

int32_t dot_u8_i8_neon(const uint8_t* x_u8, const int8_t* w, int count) {
    const int8_t* x = reinterpret_cast<const int8_t*>(x_u8);

#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    int i = 0;
    for (; i + 63 < count; i += 64) {
        acc0 = vdotq_s32(acc0, vld1q_s8(x + i), vld1q_s8(w + i));
        acc1 = vdotq_s32(acc1, vld1q_s8(x + i + 16), vld1q_s8(w + i + 16));
        acc2 = vdotq_s32(acc2, vld1q_s8(x + i + 32), vld1q_s8(w + i + 32));
        acc3 = vdotq_s32(acc3, vld1q_s8(x + i + 48), vld1q_s8(w + i + 48));
    }

    acc0 = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));

    for (; i < count; i += 16) {
        acc0 = vdotq_s32(acc0, vld1q_s8(x + i), vld1q_s8(w + i));
    }
    return vaddvq_s32(acc0);
#else
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    auto accumulate = [](int32x4_t acc, int8x16_t xv, int8x16_t wv) {
        const int16x8_t lo = vmull_s8(vget_low_s8(xv), vget_low_s8(wv));
        const int16x8_t hi = vmull_s8(vget_high_s8(xv), vget_high_s8(wv));
        const int32x4_t lo32 = vpaddlq_s16(lo);
        const int32x4_t hi32 = vpaddlq_s16(hi);
        return vaddq_s32(acc, vaddq_s32(lo32, hi32));
    };

    int i = 0;
    for (; i + 63 < count; i += 64) {
        acc0 = accumulate(acc0, vld1q_s8(x + i), vld1q_s8(w + i));
        acc1 = accumulate(acc1, vld1q_s8(x + i + 16), vld1q_s8(w + i + 16));
        acc2 = accumulate(acc2, vld1q_s8(x + i + 32), vld1q_s8(w + i + 32));
        acc3 = accumulate(acc3, vld1q_s8(x + i + 48), vld1q_s8(w + i + 48));
    }

    acc0 = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));

    for (; i < count; i += 16) {
        const int8x16_t xv = vld1q_s8(x + i);
        const int8x16_t wv = vld1q_s8(w + i);
        acc0 = accumulate(acc0, xv, wv);
    }
    return vaddvq_s32(acc0);
#endif
}
#endif

void pack_clipped_u8(uint8_t* out,
                     const int16_t* us,
                     const int16_t* them,
                     int ft_scale_i) {
#if defined(USE_AVX2)
    pack_clipped_u8_avx2(out, us, them, ft_scale_i);
#elif defined(USE_NEON)
    pack_clipped_u8_neon(out, us, them, ft_scale_i);
#else
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        int32_t a = us[i];
        int32_t b = them[i];
        if (a < 0) a = 0;
        if (b < 0) b = 0;
        if (a > ft_scale_i) a = ft_scale_i;
        if (b > ft_scale_i) b = ft_scale_i;
        out[i] = static_cast<uint8_t>(a);
        out[kExpectedFtSize + i] = static_cast<uint8_t>(b);
    }
#endif
}

int32_t dot_u8_i8(const uint8_t* x, const int8_t* w, int count) {
#if defined(USE_AVX2)
    if (count % 32 == 0) {
        return dot_u8_i8_avx2(x, w, count);
    }
#elif defined(USE_NEON)
    if (count % 16 == 0) {
        return dot_u8_i8_neon(x, w, count);
    }
#endif

    int32_t sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += static_cast<int32_t>(x[i]) * static_cast<int32_t>(w[i]);
    }
    return sum;
}

int32_t round_to_int32(double value) {
    if (value > static_cast<double>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    if (value < static_cast<double>(std::numeric_limits<int32_t>::min())) {
        return std::numeric_limits<int32_t>::min();
    }

    return static_cast<int32_t>((value >= 0.0) ? std::floor(value + 0.5) : std::ceil(value - 0.5));
}

double raw_outscale_units_to_score_stm(int64_t raw, const LoadedNetwork& network) {
    return (static_cast<double>(raw) / static_cast<double>(network.out_scale)) *
           static_cast<double>(network.score_scale);
}

double raw_outscale_units_to_cp(int64_t raw, const LoadedNetwork& network) {
    return raw_outscale_units_to_score_stm(raw, network) * kCpPerStockfishScore;
}

int64_t evaluate_state_int_raw_outscale_units(const thrawn::NnueState& state,
                                              const LoadedNetwork& network,
                                              int colour_to_move) {
    alignas(64) std::array<uint8_t, kL1InputSize> x{};
    alignas(64) std::array<uint8_t, kExpectedL1Size> h1{};
    alignas(64) std::array<uint8_t, kExpectedL2Size> h2{};

    const auto& us = (colour_to_move == white) ? state.white_acc : state.black_acc;
    const auto& them = (colour_to_move == white) ? state.black_acc : state.white_acc;

    pack_clipped_u8(x.data(), us.data(), them.data(), network.qa);

    for (int j = 0; j < static_cast<int>(kExpectedL1Size); ++j) {
        const int32_t dot = dot_u8_i8(x.data(), network.l1_weight_t[j].data(), kL1InputSize);
        int64_t s = static_cast<int64_t>(network.l1_bias[j]) +
                    div_round_by_scale(dot, network.qa);
        if (s < 0) s = 0;
        if (s > network.q1) s = network.q1;
        h1[j] = static_cast<uint8_t>(s);
    }

    for (int j = 0; j < static_cast<int>(kExpectedL2Size); ++j) {
        const int32_t dot = dot_u8_i8(h1.data(), network.l2_weight_t[j].data(), kExpectedL1Size);
        int64_t s = static_cast<int64_t>(network.l2_bias[j]) +
                    div_round_by_scale(dot, network.q1);
        if (s < 0) s = 0;
        if (s > network.q2) s = network.q2;
        h2[j] = static_cast<uint8_t>(s);
    }

    const int32_t dot = dot_u8_i8(h2.data(), network.out_weight.data(), kExpectedL2Size);
    const int64_t out = static_cast<int64_t>(network.out_bias) +
                        div_round_by_scale(dot, network.q2);

    return out;
}

int32_t evaluate_state_int_score_stm(const thrawn::NnueState& state,
                                     const LoadedNetwork& network,
                                     int colour_to_move) {
    const int64_t raw = evaluate_state_int_raw_outscale_units(state, network, colour_to_move);
    if (!std::isfinite(network.out_scale) || network.out_scale <= 0.0f ||
        !std::isfinite(network.score_scale) || network.score_scale <= 0.0f) {
        return 0;
    }

    return round_to_int32(raw_outscale_units_to_score_stm(raw, network));
}

int32_t evaluate_state_int_cp(const thrawn::NnueState& state,
                              const LoadedNetwork& network,
                              int colour_to_move) {
    const int64_t raw = evaluate_state_int_raw_outscale_units(state, network, colour_to_move);
    if (!std::isfinite(network.out_scale) || network.out_scale <= 0.0f ||
        !std::isfinite(network.score_scale) || network.score_scale <= 0.0f) {
        return 0;
    }

    return round_to_int32(raw_outscale_units_to_cp(raw, network));
}

int32_t evaluate_state_int_engine_score(const thrawn::NnueState& state,
                                        const LoadedNetwork& network,
                                        int colour_to_move) {
    if constexpr (kSearchUsesCentipawns) {
        return evaluate_state_int_cp(state, network, colour_to_move);
    } else {
        return evaluate_state_int_score_stm(state, network, colour_to_move);
    }
}

float evaluate_state_float_raw_output(const thrawn::NnueState& state,
                                      const LoadedNetwork& network,
                                      int colour_to_move) {
    std::array<float, kL1InputSize> x{};
    std::array<float, kExpectedL1Size> h1{};
    std::array<float, kExpectedL2Size> h2{};

    const auto& us = (colour_to_move == white) ? state.white_acc : state.black_acc;
    const auto& them = (colour_to_move == white) ? state.black_acc : state.white_acc;

    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        x[i] = std::clamp(static_cast<float>(us[i]) / network.ft_scale, 0.0f, 1.0f);
        x[static_cast<int>(kExpectedFtSize) + i] = std::clamp(static_cast<float>(them[i]) / network.ft_scale, 0.0f, 1.0f);
    }

    for (int j = 0; j < static_cast<int>(kExpectedL1Size); ++j) {
        float s = static_cast<float>(network.l1_bias[j]) / network.l1_scale;
        for (int i = 0; i < kL1InputSize; ++i) {
            s += x[i] * (static_cast<float>(network.l1_weight_t[j][i]) / network.l1_scale);
        }
        h1[j] = std::clamp(s, 0.0f, 1.0f);
    }

    for (int j = 0; j < static_cast<int>(kExpectedL2Size); ++j) {
        float s = static_cast<float>(network.l2_bias[j]) / network.l2_scale;
        for (int i = 0; i < static_cast<int>(kExpectedL1Size); ++i) {
            s += h1[i] * (static_cast<float>(network.l2_weight_t[j][i]) / network.l2_scale);
        }
        h2[j] = std::clamp(s, 0.0f, 1.0f);
    }

    float out = static_cast<float>(network.out_bias) / network.out_scale;
    for (int i = 0; i < static_cast<int>(kExpectedL2Size); ++i) {
        out += h2[i] * (static_cast<float>(network.out_weight[i]) / network.out_scale);
    }

    return out;
}

float evaluate_state_float_score_stm(const thrawn::NnueState& state,
                                     const LoadedNetwork& network,
                                     int colour_to_move) {
    return evaluate_state_float_raw_output(state, network, colour_to_move) * network.score_scale;
}

float evaluate_state_float_cp(const thrawn::NnueState& state,
                              const LoadedNetwork& network,
                              int colour_to_move) {
    return evaluate_state_float_score_stm(state, network, colour_to_move) *
           static_cast<float>(kCpPerStockfishScore);
}

bool verify_state_against_board(const thrawn::Position* pos,
                                const LoadedNetwork& network,
                                std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    if (pos->ply < 0 || pos->ply > MAX_DEPTH) {
        return fail("invalid ply");
    }

    thrawn::NnueState rebuilt;
    if (!build_state_from_board(pos, network, rebuilt)) {
        return fail("failed full refresh");
    }

    const thrawn::NnueState& incremental = pos->nnue_stack[pos->ply];
    if (!incremental.valid) {
        return fail("incremental state invalid");
    }

    if (incremental.piece_count != rebuilt.piece_count) {
        return fail("piece count mismatch");
    }
    if (incremental.white_king_sq != rebuilt.white_king_sq ||
        incremental.black_king_sq != rebuilt.black_king_sq) {
        return fail("king square mismatch");
    }
    if (incremental.white_acc != rebuilt.white_acc) {
        return fail("white accumulator mismatch");
    }
    if (incremental.black_acc != rebuilt.black_acc) {
        return fail("black accumulator mismatch");
    }

    for (int square = 0; square < BOARD_SIZE; ++square) {
        const int inc_index = incremental.index_by_square[square];
        const int ref_index = rebuilt.index_by_square[square];
        if ((inc_index == -1) != (ref_index == -1)) {
            return fail("piece presence mismatch");
        }
        if (inc_index != -1 && incremental.piece_list[inc_index] != rebuilt.piece_list[ref_index]) {
            return fail("piece identity mismatch");
        }
    }

    const int64_t incremental_raw = evaluate_state_int_raw_outscale_units(incremental, network, pos->colour_to_move);
    const int64_t rebuilt_raw = evaluate_state_int_raw_outscale_units(rebuilt, network, pos->colour_to_move);
    if (incremental_raw != rebuilt_raw) {
        return fail("raw evaluation mismatch");
    }

    return true;
}

bool measure_evaluation_parity(const thrawn::Position* pos,
                               const LoadedNetwork& network,
                               float* abs_error_cp,
                               std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    thrawn::NnueState refreshed_state;
    const thrawn::NnueState* state = state_for_evaluation(pos, network, refreshed_state);
    if (state == nullptr) {
        return fail("failed to build evaluation state");
    }

    const int32_t int_cp = evaluate_state_int_cp(*state, network, pos->colour_to_move);
    const float float_cp = evaluate_state_float_cp(*state, network, pos->colour_to_move);
    if (abs_error_cp != nullptr) {
        *abs_error_cp = std::fabs(float_cp - static_cast<float>(int_cp));
    }

    return true;
}

bool run_halfkp_index_self_test(std::string& error) {
    for (const HalfkpSelfTestCase& test : kHalfkpSelfTests) {
        const int actual = halfkp_index(test.piece,
                                        test.piece_square,
                                        test.king_square,
                                        test.white_perspective);
        if (actual != test.expected_index) {
            error = "halfkp index self-test failed";
            return false;
        }
    }
    return true;
}

bool load_network_from_file(const std::string& path,
                            LoadedNetwork& network,
                            std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "could not open file";
        return false;
    }

    char magic[8] = {};
    stream.read(magic, sizeof(magic));
    if (!stream) {
        error = "file too small to contain header";
        return false;
    }
    if (std::memcmp(magic, kExpectedMagic, sizeof(magic)) != 0) {
        error = "unexpected magic";
        return false;
    }

    uint32_t version = 0;
    uint32_t num_features = 0;
    uint32_t ft_size = 0;
    uint32_t l1_size = 0;
    uint32_t l2_size = 0;
    uint32_t output_perspective = 0;

    float ft_scale = 0.0f;
    float l1_scale = 0.0f;
    float l2_scale = 0.0f;
    float out_scale = 0.0f;
    float score_scale = 0.0f;

    uint32_t description_length = 0;

    if (!read_scalar_le(stream, version)) {
        error = "failed while reading version";
        return false;
    }

    const std::string feature_set = read_feature_set(stream);

    if (!stream ||
        !read_scalar_le(stream, num_features) ||
        !read_scalar_le(stream, ft_size) ||
        !read_scalar_le(stream, l1_size) ||
        !read_scalar_le(stream, l2_size) ||
        !read_scalar_le(stream, output_perspective) ||
        !read_scalar_le(stream, ft_scale) ||
        !read_scalar_le(stream, l1_scale) ||
        !read_scalar_le(stream, l2_scale) ||
        !read_scalar_le(stream, out_scale) ||
        !read_scalar_le(stream, score_scale) ||
        !read_scalar_le(stream, description_length)) {
        error = "failed while reading header";
        return false;
    }

    if (version != kCurrentVersion) {
        error = "unsupported version";
        return false;
    }
    if (feature_set != kExpectedFeatureSet) {
        error = "unexpected feature set";
        return false;
    }
    if (num_features != kExpectedNumFeatures || ft_size != kExpectedFtSize ||
        l1_size != kExpectedL1Size || l2_size != kExpectedL2Size) {
        error = "unexpected network dimensions";
        return false;
    }
    if (output_perspective != kExpectedOutputPerspective) {
        error = "unexpected output perspective";
        return false;
    }

    int32_t qa = 0;
    int32_t q1 = 0;
    int32_t q2 = 0;

    if (!parse_header_scale(ft_scale, "ft_scale", kFtActivationMax, qa, error) ||
        !parse_header_scale(l1_scale, "l1_scale", kDenseActivationMax, q1, error) ||
        !parse_header_scale(l2_scale, "l2_scale", kDenseActivationMax, q2, error)) {
        return false;
    }
    if (!parse_positive_bounded_float(out_scale, "out_scale", static_cast<float>(kDenseActivationMax), error) ||
        !parse_positive_bounded_float(score_scale, "score_scale", std::numeric_limits<float>::max(), error)) {
        return false;
    }

    std::string description(description_length, '\0');
    if (description_length > 0) {
        stream.read(description.data(), static_cast<std::streamsize>(description_length));
        if (!stream) {
            error = "failed while reading description";
            return false;
        }
    }

    std::vector<int16_t> ft_bias_q;
    std::vector<int16_t> ft_weight_q;

    std::vector<int32_t> l1_bias_q;
    std::vector<int8_t> l1_weight_q;

    std::vector<int32_t> l2_bias_q;
    std::vector<int8_t> l2_weight_q;

    std::vector<int32_t> out_bias_q;
    std::vector<int8_t> out_weight_q;

    if (!read_vector_le(stream, ft_bias_q, kExpectedFtSize) ||
        !read_vector_le(stream, ft_weight_q, static_cast<std::size_t>(kExpectedNumFeatures) * kExpectedFtSize) ||
        !read_vector_le(stream, l1_bias_q, kExpectedL1Size) ||
        !read_vector_le(stream, l1_weight_q, static_cast<std::size_t>(kL1InputSize) * kExpectedL1Size) ||
        !read_vector_le(stream, l2_bias_q, kExpectedL2Size) ||
        !read_vector_le(stream, l2_weight_q, static_cast<std::size_t>(kExpectedL1Size) * kExpectedL2Size) ||
        !read_vector_le(stream, out_bias_q, 1) ||
        !read_vector_le(stream, out_weight_q, kExpectedL2Size)) {
        error = "failed while reading tensors";
        return false;
    }

    if (!stream) {
        error = "unexpected end of file";
        return false;
    }
    if (stream.peek() != std::ifstream::traits_type::eof()) {
        error = "unexpected trailing data";
        return false;
    }

    network = LoadedNetwork{};
    network.version = version;
    network.output_perspective = output_perspective;

    network.ft_scale = ft_scale;
    network.l1_scale = l1_scale;
    network.l2_scale = l2_scale;
    network.out_scale = out_scale;
    network.score_scale = score_scale;

    network.qa = qa;
    network.q1 = q1;
    network.q2 = q2;

    std::copy(ft_bias_q.begin(), ft_bias_q.end(), network.ft_bias.begin());
    network.ft_weight = std::move(ft_weight_q);

    std::copy(l1_bias_q.begin(), l1_bias_q.end(), network.l1_bias.begin());
    std::copy(l2_bias_q.begin(), l2_bias_q.end(), network.l2_bias.begin());

    for (int j = 0; j < static_cast<int>(kExpectedL1Size); ++j) {
        for (int i = 0; i < kL1InputSize; ++i) {
            network.l1_weight_t[j][i] = l1_weight_q[static_cast<std::size_t>(i) * kExpectedL1Size + j];
        }
    }

    for (int j = 0; j < static_cast<int>(kExpectedL2Size); ++j) {
        for (int i = 0; i < static_cast<int>(kExpectedL1Size); ++i) {
            network.l2_weight_t[j][i] = l2_weight_q[static_cast<std::size_t>(i) * kExpectedL2Size + j];
        }
    }

    network.out_bias = out_bias_q[0];
    std::copy(out_weight_q.begin(), out_weight_q.end(), network.out_weight.begin());

    network.description = std::move(description);
    network.loaded_path = path;

    if (!run_halfkp_index_self_test(error)) {
        return false;
    }

    return true;
}

#ifdef DEBUG_BUILD
[[noreturn]] void parity_failure(const char* message, const thrawn::Position* pos) {
    std::cerr << "NNUE parity failure: " << message
              << " ply=" << pos->ply
              << " side=" << pos->colour_to_move
              << std::endl;
    std::abort();
}
#endif

} // namespace

void nnue_init(const char* evalFile) {
    const std::string requested = evalFile == nullptr ? "" : std::string(evalFile);
    std::string last_error;
    auto candidates = candidate_paths(requested);

    for (const std::string& candidate : candidates) {
        LoadedNetwork loaded;
        std::string error;
        if (!load_network_from_file(candidate, loaded, error)) {
            if (last_error.empty() || error != "could not open file") {
                last_error = candidate + ": " + error;
            }
            continue;
        }

        auto network = std::make_shared<LoadedNetwork>(std::move(loaded));
        {
            std::lock_guard<std::mutex> lock(g_network_mutex);
            g_network_owner = network;
            g_network_raw.store(network.get(), std::memory_order_release);
        }

        std::cout << "info string Loaded NNUE from " << candidate << "\n";
        return;
    }

    {
        std::lock_guard<std::mutex> lock(g_network_mutex);
        g_network_owner.reset();
        g_network_raw.store(nullptr, std::memory_order_release);
    }

    std::cout << "info string Failed to load NNUE";
    if (!last_error.empty()) {
        std::cout << " (" << last_error << ")";
    }
    std::cout << "\n";
}

bool nnue_loaded() {
    return current_network() != nullptr;
}

int nnue_evaluate(thrawn::Position* pos) {
    const LoadedNetwork* network = current_network();
    if (network == nullptr) {
        return 0;
    }

    thrawn::NnueState refreshed_state;
    const thrawn::NnueState* state = state_for_evaluation(pos, *network, refreshed_state);
    if (state == nullptr) {
        return 0;
    }

    return evaluate_state_int_engine_score(*state, *network, pos->colour_to_move);
}

float nnue_evaluate_raw(const thrawn::Position* pos) {
    const LoadedNetwork* network = current_network();
    if (network == nullptr) {
        return 0.0f;
    }

    thrawn::NnueState refreshed_state;
    const thrawn::NnueState* state = state_for_evaluation(pos, *network, refreshed_state);
    if (state == nullptr) {
        return 0.0f;
    }

    return evaluate_state_float_score_stm(*state, *network, pos->colour_to_move);
}

void nnue_refresh_root(thrawn::Position* pos) {
    thrawn::NnueState& root = pos->nnue_stack[0];
    const LoadedNetwork* network = current_network();
    if (network == nullptr) {
        clear_state(root);
        return;
    }

    const bool refreshed = build_state_from_board(pos, *network, root);
#ifdef DEBUG_BUILD
    if (refreshed) {
        nnue_debug_check(pos);
    }
#endif
}

void nnue_copy_parent_to_child(thrawn::Position* pos, int child_ply) {
    if (child_ply <= 0 || child_ply > MAX_DEPTH) {
        return;
    }
    pos->nnue_stack[child_ply] = pos->nnue_stack[child_ply - 1];
}

void nnue_add_piece(thrawn::Position* pos, int ply, int piece, int square) {
    const LoadedNetwork* network = current_network();
    if (network == nullptr || ply < 0 || ply > MAX_DEPTH) {
        return;
    }

    thrawn::NnueState& state = pos->nnue_stack[ply];
    if (!state.valid) {
        return;
    }

    if (!add_piece_to_state(state, *network, piece, square)) {
        state.valid = false;
        return;
    }

    if (piece == K) {
        if (!refresh_perspective_accumulator_from_board(pos, state, *network, true)) {
            state.valid = false;
        }
    } else if (piece == k) {
        if (!refresh_perspective_accumulator_from_board(pos, state, *network, false)) {
            state.valid = false;
        }
    }
}

void nnue_remove_piece(thrawn::Position* pos, int ply, int piece, int square) {
    const LoadedNetwork* network = current_network();
    if (network == nullptr || ply < 0 || ply > MAX_DEPTH) {
        return;
    }

    thrawn::NnueState& state = pos->nnue_stack[ply];
    if (!state.valid) {
        return;
    }

    if (!remove_piece_from_state(state, *network, piece, square)) {
        state.valid = false;
    }
}

bool nnue_verify_position(const thrawn::Position* pos, std::string* error) {
    const LoadedNetwork* network = current_network();
    if (network == nullptr) {
        if (error != nullptr) {
            *error = "NNUE not loaded";
        }
        return false;
    }
    return verify_state_against_board(pos, *network, error);
}

bool nnue_measure_evaluation_parity(const thrawn::Position* pos,
                                    float* abs_error_cp,
                                    std::string* error) {
    const LoadedNetwork* network = current_network();
    if (network == nullptr) {
        if (error != nullptr) {
            *error = "NNUE not loaded";
        }
        return false;
    }
    return measure_evaluation_parity(pos, *network, abs_error_cp, error);
}

void nnue_debug_check(const thrawn::Position* pos) {
#ifndef DEBUG_BUILD
    (void)pos;
#else
    std::string error;
    if (!nnue_verify_position(pos, &error)) {
        parity_failure(error.c_str(), pos);
    }
#endif
}
