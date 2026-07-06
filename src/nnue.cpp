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
#include <new>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

#if defined(USE_NEON)
#include <arm_neon.h>
#endif

namespace {

constexpr char kExpectedMagic[8] = {'T', 'H', 'N', 'N', 'U', 'E', '\0', '\1'};
constexpr uint32_t kCurrentVersion = 9;
constexpr uint32_t kExpectedNumFeatures = thrawn::NNUE_INPUT_FEATURES;
constexpr uint32_t kExpectedPsNb = thrawn::NNUE_PS_NB;
constexpr uint32_t kExpectedFtSize = thrawn::NNUE_FT_SIZE;
constexpr uint32_t kExpectedHiddenSize = thrawn::NNUE_HIDDEN_SIZE;
constexpr uint32_t kExpectedForwardSize = thrawn::NNUE_FORWARD_SIZE;
constexpr uint32_t kExpectedFc0OutputSize = thrawn::NNUE_FC0_OUTPUT_SIZE;
constexpr uint32_t kExpectedFc0InputSize = thrawn::NNUE_FT_SIZE;   // pairwise SqrCReLU: 2*FtSize -> FtSize
constexpr uint32_t kExpectedFc1InputSize = thrawn::NNUE_FC1_INPUT_SIZE;
constexpr uint32_t kExpectedFc1OutputSize = thrawn::NNUE_FC1_OUTPUT_SIZE;
constexpr uint32_t kExpectedOutputPerspective = 1;
constexpr const char* kExpectedFeatureSet = "HalfKAv2_hm";
constexpr const char* kDefaultEvalFile = "thrawn-nn-2.nnue";

// The FT output is activated (pairwise SqrCReLU) before fc0, so fc0 is a
// u8 x i8 layer whose input width equals the accumulator width, not twice it.
constexpr int kFc0InputSize = static_cast<int>(kExpectedFtSize);
constexpr int kFtActivationHalf = static_cast<int>(kExpectedFtSize) / 2;
constexpr int kFc1InputPaddedSize = 64;
constexpr int kFc2OutputSize = 1;
constexpr int kKingPlaneOffset = 10 * 64;
constexpr double kCpPerStockfishScore = 100.0 / 208.0;
constexpr int8_t kLocalAccumulatorSource = -1;

// Thrawn's search margins and mate bounds are cp-based, so the NNUE score_stm
// is converted once at the engine boundary. UCI raw helpers still expose score_stm.
constexpr bool kSearchUsesCentipawns = true;

static_assert(kExpectedFc0OutputSize == kExpectedHiddenSize + kExpectedForwardSize,
              "HalfKAv2_hm fc0 shape mismatch");
static_assert(kExpectedFc1InputSize == kExpectedHiddenSize * 2,
              "HalfKAv2_hm fc1 input mismatch");
static_assert(kExpectedFc0OutputSize % 4 == 0, "fc0 output must fit x4 dense kernels");
static_assert(kExpectedFc1OutputSize % 4 == 0, "fc1 output must fit x4 dense kernels");
static_assert(kFc1InputPaddedSize >= static_cast<int>(kExpectedFc1InputSize),
              "fc1 padding too small");
static_assert(kFc1InputPaddedSize % 32 == 0, "fc1 padded input must fit AVX2 kernels");
static_assert(kFc0InputSize % 64 == 0, "fc0 u8 x i8 dense kernels consume 64 inputs per step");
static_assert(kFtActivationHalf % 32 == 0, "pairwise SqrCReLU processes 32 lanes per step");

constexpr std::array<int, 64> kKingBuckets{{
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
     8,  9, 10, 11, 11, 10,  9,  8,
     4,  5,  6,  7,  7,  6,  5,  4,
     0,  1,  2,  3,  3,  2,  1,  0,
}};

struct HalfKAv2SelfTestCase {
    int piece;
    int piece_square;
    int king_square;
    bool white_perspective;
    int expected_index;
};

constexpr std::array<HalfKAv2SelfTestCase, 5> kHalfKAv2SelfTests{{
    {P, a2, e1, true, 21832},
    {p, a7, e1, true, 21936},
    {R, h1, e8, false, 22335},
    {k, e8, e1, true, 22524},
    {Q, a1, d1, true, 22343},
}};

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        void* ptr = nullptr;
        const std::size_t bytes = count * sizeof(T);
#if defined(_WIN32)
        ptr = _aligned_malloc(bytes, Alignment);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) noexcept {
    return false;
}

using AlignedFtWeights = std::vector<int16_t, AlignedAllocator<int16_t, thrawn::NNUE_SIMD_ALIGNMENT>>;

struct AccumulatorView {
    const std::array<int16_t, kExpectedFtSize>* white = nullptr;
    const std::array<int16_t, kExpectedFtSize>* black = nullptr;

    bool valid() const {
        return white != nullptr && black != nullptr;
    }
};

struct EvaluationState {
    const thrawn::NnueState* state = nullptr;
    AccumulatorView accumulators{};

    bool valid() const {
        return state != nullptr && accumulators.valid();
    }
};

struct IntEvaluationResult {
    int32_t fc2 = 0;
    int32_t forward = 0;

    bool operator==(const IntEvaluationResult& other) const {
        return fc2 == other.fc2 && forward == other.forward;
    }
};

struct LoadedNetwork {
    uint32_t version = 0;
    uint32_t output_perspective = 0;

    float ft_scale = 0.0f;
    float fc0_scale = 0.0f;
    float fc1_scale = 0.0f;
    float fc2_scale = 0.0f;
    float score_scale = 0.0f;

    int32_t qft = 0;
    int32_t qfc0 = 0;
    int32_t qfc1 = 0;
    int32_t qfc2 = 0;

    // Pairwise SqrCReLU realization of the FT activation. `ft_one` is the int16
    // value representing 1.0 (== qft). The clamped product is renormalized by
    // `>> act_shift` so its "one" is `act_one` (a uint8, e.g. 127 for ft_one=255).
    int32_t ft_one = 0;
    int32_t act_shift = 0;
    int32_t act_one = 0;

    float inv_ft_scale = 0.0f;
    float inv_fc0_scale = 0.0f;
    float inv_fc1_scale = 0.0f;
    float inv_fc2_scale = 0.0f;

    std::array<int16_t, kExpectedFtSize> ft_bias{};
    AlignedFtWeights ft_weight;

    std::array<int32_t, kExpectedFc0OutputSize> fc0_bias{};
    alignas(64) std::array<std::array<int8_t, kFc0InputSize>, kExpectedFc0OutputSize> fc0_weight_t{};

    std::array<int32_t, kExpectedFc1OutputSize> fc1_bias{};
    alignas(64) std::array<std::array<int8_t, kFc1InputPaddedSize>, kExpectedFc1OutputSize> fc1_weight_t{};

    int32_t fc2_bias = 0;
    alignas(64) std::array<int8_t, kExpectedFc1OutputSize> fc2_weight{};

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

template <typename T, typename Allocator>
bool read_vector_le(std::ifstream& stream,
                    std::vector<T, Allocator>& values,
                    std::size_t count) {
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
        path = kDefaultEvalFile;
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

bool parse_integer_scale(float scale,
                         const char* scale_name,
                         int32_t max_allowed,
                         int32_t& out_scale,
                         std::string& error) {
    if (!std::isfinite(scale) || scale <= 0.0f) {
        error = std::string("invalid ") + scale_name;
        return false;
    }

    const double rounded = std::round(static_cast<double>(scale));
    if (std::fabs(static_cast<double>(scale) - rounded) > 1e-3 ||
        rounded <= 0.0 ||
        rounded > static_cast<double>(max_allowed)) {
        error = std::string(scale_name) + " must be a positive integer scale";
        return false;
    }

    out_scale = static_cast<int32_t>(rounded);
    return true;
}

int64_t div_round_by_scale(int64_t value, int32_t scale) {
    if (scale <= 0) {
        return 0;
    }

    const uint64_t magnitude = value >= 0
        ? static_cast<uint64_t>(value)
        : static_cast<uint64_t>(-value);
    const uint64_t quotient = (magnitude + static_cast<uint32_t>(scale) / 2U) /
                              static_cast<uint32_t>(scale);
    const int64_t signed_quotient = static_cast<int64_t>(quotient);
    return value >= 0 ? signed_quotient : -signed_quotient;
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

int engine_to_model_square(int square) {
    return square ^ 56;
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

int piece_square_offset(int piece, bool white_perspective) {
    if (is_king_piece(piece)) {
        return kKingPlaneOffset;
    }

    const int ptype = piece_type_index(piece);
    if (ptype < 0) {
        return -1;
    }

    const bool friendly = white_perspective == is_white_piece(piece);
    return (ptype * 2 + (friendly ? 0 : 1)) * 64;
}

int halfkav2_hm_index(int piece,
                      int piece_square,
                      int our_king_square,
                      bool white_perspective) {
    if (piece_square < 0 || piece_square >= BOARD_SIZE ||
        our_king_square < 0 || our_king_square >= BOARD_SIZE) {
        return -1;
    }

    const int offset = piece_square_offset(piece, white_perspective);
    if (offset < 0) {
        return -1;
    }

    const int model_piece_sq = engine_to_model_square(piece_square);
    const int model_king_sq = engine_to_model_square(our_king_square);
    const int flip = white_perspective ? 0 : 56;
    const int orient = (model_king_sq & 7) < 4 ? 7 : 0;
    const int oriented_sq = model_piece_sq ^ orient ^ flip;
    const int king_bucket = kKingBuckets[model_king_sq ^ flip];

    const int index = oriented_sq + offset + king_bucket * static_cast<int>(kExpectedPsNb);
    return (index >= 0 && index < static_cast<int>(kExpectedNumFeatures)) ? index : -1;
}

bool should_patch_piece_for_perspective(int piece, bool white_perspective) {
    if (!is_king_piece(piece)) {
        return true;
    }

    return (piece == K) ? !white_perspective : white_perspective;
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
    state.white_acc_source_ply = kLocalAccumulatorSource;
    state.black_acc_source_ply = kLocalAccumulatorSource;
    state.piece_count = 0;
    state.white_king_sq = -1;
    state.black_king_sq = -1;
    state.valid = false;
}

AccumulatorView local_accumulator_view(const thrawn::NnueState& state) {
    return {&state.white_acc, &state.black_acc};
}

int accumulator_source_for_child(const thrawn::NnueState& parent,
                                 bool white_perspective,
                                 int parent_ply) {
    const int parent_source = white_perspective
        ? parent.white_acc_source_ply
        : parent.black_acc_source_ply;
    return parent_source == kLocalAccumulatorSource ? parent_ply : parent_source;
}

bool resolve_accumulator(const thrawn::Position* pos,
                         int ply,
                         bool white_perspective,
                         const std::array<int16_t, kExpectedFtSize>*& accumulator) {
    if (pos == nullptr || ply < 0 || ply > MAX_DEPTH) {
        return false;
    }

    int source_ply = ply;
    for (int hops = 0; hops <= MAX_DEPTH; ++hops) {
        const thrawn::NnueState& state = pos->nnue_stack[static_cast<std::size_t>(source_ply)];
        if (!state.valid) {
            return false;
        }

        const int next_source = white_perspective
            ? state.white_acc_source_ply
            : state.black_acc_source_ply;
        if (next_source == kLocalAccumulatorSource) {
            accumulator = white_perspective ? &state.white_acc : &state.black_acc;
            return true;
        }
        if (next_source < 0 || next_source > MAX_DEPTH || next_source == source_ply) {
            return false;
        }
        source_ply = next_source;
    }

    return false;
}

bool accumulator_view_for_ply(const thrawn::Position* pos,
                              int ply,
                              AccumulatorView& view) {
    return resolve_accumulator(pos, ply, true, view.white) &&
           resolve_accumulator(pos, ply, false, view.black);
}

bool materialize_accumulator(thrawn::Position* pos, int ply, bool white_perspective) {
    if (pos == nullptr || ply < 0 || ply > MAX_DEPTH) {
        return false;
    }

    thrawn::NnueState& state = pos->nnue_stack[static_cast<std::size_t>(ply)];
    int8_t& source = white_perspective ? state.white_acc_source_ply : state.black_acc_source_ply;
    if (source == kLocalAccumulatorSource) {
        return state.valid;
    }

    const std::array<int16_t, kExpectedFtSize>* source_acc = nullptr;
    if (!resolve_accumulator(pos, ply, white_perspective, source_acc) || source_acc == nullptr) {
        return false;
    }

    auto& target = white_perspective ? state.white_acc : state.black_acc;
    if (source_acc != &target) {
        target = *source_acc;
    }
    source = kLocalAccumulatorSource;
    return true;
}

bool materialize_state_accumulators(thrawn::Position* pos, int ply) {
    return materialize_accumulator(pos, ply, true) &&
           materialize_accumulator(pos, ply, false);
}

void mark_accumulator_local(thrawn::NnueState& state, bool white_perspective) {
    if (white_perspective) {
        state.white_acc_source_ply = kLocalAccumulatorSource;
    } else {
        state.black_acc_source_ply = kLocalAccumulatorSource;
    }
}

void copy_state_metadata_without_accumulators(thrawn::NnueState& child,
                                              const thrawn::NnueState& parent,
                                              int child_ply) {
    child.piece_list = parent.piece_list;
    child.square_list = parent.square_list;
    child.index_by_square = parent.index_by_square;
    child.piece_count = parent.piece_count;
    child.white_king_sq = parent.white_king_sq;
    child.black_king_sq = parent.black_king_sq;
    child.valid = parent.valid;

    const int parent_ply = child_ply - 1;
    child.white_acc_source_ply = static_cast<int8_t>(
        accumulator_source_for_child(parent, true, parent_ply));
    child.black_acc_source_ply = static_cast<int8_t>(
        accumulator_source_for_child(parent, false, parent_ply));
}

enum class AccumulatorUpdateOp { Add, Subtract };
constexpr int kMaxBatchedPieceUpdates = 8;

struct AccumulatorFeatureUpdate {
    int feature = -1;
    AccumulatorUpdateOp op = AccumulatorUpdateOp::Add;
};

#if defined(USE_AVX2)
// The 1024-int16 (2 KB) accumulator is patched on every make-move, so each kernel
// is unrolled 2x (two independent 256-bit chains) to expose ILP and hide load
// latency instead of a single load->op->store dependency per iteration. FtSize is
// a multiple of 32, so the unrolled stride divides evenly. All ops are wrapping
// (non-saturating), so the reordering/fusion below is bit-identical to the naive
// element-by-element version.
static inline void acc_add_row_avx2(int16_t* acc, const int16_t* row) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        const __m256i a0 = _mm256_add_epi16(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i)),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i)));
        const __m256i a1 = _mm256_add_epi16(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i + 16)),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i + 16)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i + 16), a1);
    }
}

static inline void acc_sub_row_avx2(int16_t* acc, const int16_t* row) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        const __m256i a0 = _mm256_sub_epi16(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i)),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i)));
        const __m256i a1 = _mm256_sub_epi16(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i + 16)),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i + 16)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i + 16), a1);
    }
}

static inline void acc_add_sub_row_avx2(int16_t* acc, const int16_t* add, const int16_t* sub) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        const __m256i a0 = _mm256_sub_epi16(
            _mm256_add_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i)),
                             _mm256_load_si256(reinterpret_cast<const __m256i*>(add + i))),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(sub + i)));
        const __m256i a1 = _mm256_sub_epi16(
            _mm256_add_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i + 16)),
                             _mm256_load_si256(reinterpret_cast<const __m256i*>(add + i + 16))),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(sub + i + 16)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i + 16), a1);
    }
}

// Fused copy+patch variants: write dst = src (+add) (-sub) in a single pass,
// reading from a separate source accumulator. Used when a child ply still shares
// an ancestor's accumulator: instead of a 2 KB memcpy(dst,src) followed by an
// in-place patch (2 reads + 2 writes of the accumulator), this does 1 read of src
// + 1 write of dst, halving accumulator memory traffic on the make-move hot path.
// Bit-identical to copy-then-patch (wrapping arithmetic). See the NEON twins.
static inline void acc_add_row_from_avx2(int16_t* dst, const int16_t* src, const int16_t* add) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        const __m256i a0 = _mm256_add_epi16(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i)),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(add + i)));
        const __m256i a1 = _mm256_add_epi16(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i + 16)),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(add + i + 16)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i + 16), a1);
    }
}

static inline void acc_sub_row_from_avx2(int16_t* dst, const int16_t* src, const int16_t* sub) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        const __m256i a0 = _mm256_sub_epi16(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i)),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(sub + i)));
        const __m256i a1 = _mm256_sub_epi16(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i + 16)),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(sub + i + 16)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i + 16), a1);
    }
}

static inline void acc_add_sub_row_from_avx2(int16_t* dst, const int16_t* src,
                                             const int16_t* add, const int16_t* sub) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        const __m256i a0 = _mm256_sub_epi16(
            _mm256_add_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(src + i)),
                             _mm256_load_si256(reinterpret_cast<const __m256i*>(add + i))),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(sub + i)));
        const __m256i a1 = _mm256_sub_epi16(
            _mm256_add_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(src + i + 16)),
                             _mm256_load_si256(reinterpret_cast<const __m256i*>(add + i + 16))),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(sub + i + 16)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i + 16), a1);
    }
}

// Pairwise SqrCReLU on one perspective's accumulator: for i in [0, half)
//   out[i] = (clamp(acc[i], 0, ft_one) * clamp(acc[i + half], 0, ft_one)) >> shift
// ft_one <= 255 keeps clamp(a)*clamp(b) < 2^16, so _mm256_mullo_epi16 yields the
// exact product and _mm256_srl_epi16 (logical) renormalizes it into [0, act_one].
void ft_pairwise_screlu_avx2(const int16_t* acc, uint8_t* out, int half, int ft_one, int shift) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i vmax = _mm256_set1_epi16(static_cast<int16_t>(ft_one));
    const __m128i vshift = _mm_cvtsi32_si128(shift);

    for (int i = 0; i < half; i += 32) {
        __m256i lo_a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i lo_b = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i + half));
        lo_a = _mm256_min_epi16(_mm256_max_epi16(lo_a, zero), vmax);
        lo_b = _mm256_min_epi16(_mm256_max_epi16(lo_b, zero), vmax);
        const __m256i lo = _mm256_srl_epi16(_mm256_mullo_epi16(lo_a, lo_b), vshift);

        __m256i hi_a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i + 16));
        __m256i hi_b = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i + 16 + half));
        hi_a = _mm256_min_epi16(_mm256_max_epi16(hi_a, zero), vmax);
        hi_b = _mm256_min_epi16(_mm256_max_epi16(hi_b, zero), vmax);
        const __m256i hi = _mm256_srl_epi16(_mm256_mullo_epi16(hi_a, hi_b), vshift);

        __m256i packed = _mm256_packus_epi16(lo, hi);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_store_si256(reinterpret_cast<__m256i*>(out + i), packed);
    }
}

#endif

#if defined(USE_NEON)
// The accumulator is 1024 int16 (16 KB) and these run on every make-move, so the
// loops are unrolled 4x (four independent 128-bit chains) to expose ILP across
// Apple/ARM's multiple NEON units instead of a single load->op->store dependency
// per iteration. All ops are wrapping (non-saturating) so the reordering/fusion
// below is bit-identical to the naive element-by-element version.
static inline void acc_add_row_neon(int16_t* acc, const int16_t* row) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        vst1q_s16(acc + i,      vaddq_s16(vld1q_s16(acc + i),      vld1q_s16(row + i)));
        vst1q_s16(acc + i + 8,  vaddq_s16(vld1q_s16(acc + i + 8),  vld1q_s16(row + i + 8)));
        vst1q_s16(acc + i + 16, vaddq_s16(vld1q_s16(acc + i + 16), vld1q_s16(row + i + 16)));
        vst1q_s16(acc + i + 24, vaddq_s16(vld1q_s16(acc + i + 24), vld1q_s16(row + i + 24)));
    }
}

static inline void acc_sub_row_neon(int16_t* acc, const int16_t* row) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 32) {
        vst1q_s16(acc + i,      vsubq_s16(vld1q_s16(acc + i),      vld1q_s16(row + i)));
        vst1q_s16(acc + i + 8,  vsubq_s16(vld1q_s16(acc + i + 8),  vld1q_s16(row + i + 8)));
        vst1q_s16(acc + i + 16, vsubq_s16(vld1q_s16(acc + i + 16), vld1q_s16(row + i + 16)));
        vst1q_s16(acc + i + 24, vsubq_s16(vld1q_s16(acc + i + 24), vld1q_s16(row + i + 24)));
    }
}

static inline void acc_add_sub_row_neon(int16_t* acc, const int16_t* add, const int16_t* sub) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
        const int16x8_t a0 = vsubq_s16(vaddq_s16(vld1q_s16(acc + i),     vld1q_s16(add + i)),     vld1q_s16(sub + i));
        const int16x8_t a1 = vsubq_s16(vaddq_s16(vld1q_s16(acc + i + 8), vld1q_s16(add + i + 8)), vld1q_s16(sub + i + 8));
        vst1q_s16(acc + i, a0);
        vst1q_s16(acc + i + 8, a1);
    }
}

// Fused copy+patch variants: write dst = src (+adds) (-subs) in a single pass,
// reading from a separate source accumulator. Used when a child ply still shares
// an ancestor's accumulator: instead of memcpy(dst,src) followed by an in-place
// patch (2 reads + 2 writes of the 16 KB accumulator), this does 1 read of src +
// 1 write of dst, halving accumulator memory traffic on the make-move hot path.
static inline void acc_add_sub_row_from_neon(int16_t* dst, const int16_t* src,
                                             const int16_t* add, const int16_t* sub) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
        const int16x8_t a0 = vsubq_s16(vaddq_s16(vld1q_s16(src + i),     vld1q_s16(add + i)),     vld1q_s16(sub + i));
        const int16x8_t a1 = vsubq_s16(vaddq_s16(vld1q_s16(src + i + 8), vld1q_s16(add + i + 8)), vld1q_s16(sub + i + 8));
        vst1q_s16(dst + i, a0);
        vst1q_s16(dst + i + 8, a1);
    }
}

static inline void acc_add_row_from_neon(int16_t* dst, const int16_t* src, const int16_t* add) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
        vst1q_s16(dst + i,     vaddq_s16(vld1q_s16(src + i),     vld1q_s16(add + i)));
        vst1q_s16(dst + i + 8, vaddq_s16(vld1q_s16(src + i + 8), vld1q_s16(add + i + 8)));
    }
}

static inline void acc_sub_row_from_neon(int16_t* dst, const int16_t* src, const int16_t* sub) {
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
        vst1q_s16(dst + i,     vsubq_s16(vld1q_s16(src + i),     vld1q_s16(sub + i)));
        vst1q_s16(dst + i + 8, vsubq_s16(vld1q_s16(src + i + 8), vld1q_s16(sub + i + 8)));
    }
}

// Pairwise SqrCReLU (see the AVX2 twin for the math). vmulq_s16 keeps the exact
// low 16 bits (ft_one <= 255 so the product stays below 2^16); vshlq_u16 by a
// negative count is a per-lane logical right shift, matching the scalar >>.
void ft_pairwise_screlu_neon(const int16_t* acc, uint8_t* out, int half, int ft_one, int shift) {
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t vmax = vdupq_n_s16(static_cast<int16_t>(ft_one));
    const int16x8_t vneg_shift = vdupq_n_s16(static_cast<int16_t>(-shift));

    for (int i = 0; i < half; i += 16) {
        int16x8_t a0 = vminq_s16(vmaxq_s16(vld1q_s16(acc + i), zero), vmax);
        int16x8_t b0 = vminq_s16(vmaxq_s16(vld1q_s16(acc + i + half), zero), vmax);
        uint16x8_t p0 = vshlq_u16(vreinterpretq_u16_s16(vmulq_s16(a0, b0)), vneg_shift);

        int16x8_t a1 = vminq_s16(vmaxq_s16(vld1q_s16(acc + i + 8), zero), vmax);
        int16x8_t b1 = vminq_s16(vmaxq_s16(vld1q_s16(acc + i + 8 + half), zero), vmax);
        uint16x8_t p1 = vshlq_u16(vreinterpretq_u16_s16(vmulq_s16(a1, b1)), vneg_shift);

        vst1q_u8(out + i, vcombine_u8(vqmovn_u16(p0), vqmovn_u16(p1)));
    }
}

#endif

bool update_accumulator_row(std::array<int16_t, kExpectedFtSize>& accumulator,
                            const AlignedFtWeights& weights,
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

bool update_accumulator_rows(std::array<int16_t, kExpectedFtSize>& accumulator,
                             const AlignedFtWeights& weights,
                             const AccumulatorFeatureUpdate* updates,
                             int update_count);

// Compute dst = src (+ add rows) (- sub rows) in a single pass. When src and dst
// alias (already-local accumulator) this is an in-place patch; when they differ
// (child ply still sharing an ancestor's accumulator) it fuses the materialize
// copy into the patch, so the 16 KB accumulator is read once and written once
// instead of copied and then re-read/re-written. Bit-identical to a plain copy
// followed by update_accumulator_rows since every op is wrapping arithmetic.
bool update_accumulator_rows_from(std::array<int16_t, kExpectedFtSize>& dst,
                                  const std::array<int16_t, kExpectedFtSize>& src,
                                  const AlignedFtWeights& weights,
                                  const AccumulatorFeatureUpdate* updates,
                                  int update_count) {
    if (update_count <= 0) {
        if (&dst != &src) {
            dst = src;
        }
        return true;
    }
    if (updates == nullptr || update_count > kMaxBatchedPieceUpdates) {
        return false;
    }

    const int16_t* add_rows[kMaxBatchedPieceUpdates];
    const int16_t* sub_rows[kMaxBatchedPieceUpdates];
    int add_count = 0;
    int sub_count = 0;

    for (int i = 0; i < update_count; ++i) {
        const int feature = updates[i].feature;
        if (feature < 0) {
            return false;
        }

        const std::size_t offset = static_cast<std::size_t>(feature) * kExpectedFtSize;
        if (offset + kExpectedFtSize > weights.size()) {
            return false;
        }

        if (updates[i].op == AccumulatorUpdateOp::Add) {
            add_rows[add_count++] = weights.data() + offset;
        } else {
            sub_rows[sub_count++] = weights.data() + offset;
        }
    }

    int16_t* d = dst.data();
    const int16_t* s = src.data();
#if defined(USE_AVX2)
    if (add_count == 1 && sub_count == 0) {
        acc_add_row_from_avx2(d, s, add_rows[0]);
        return true;
    }
    if (add_count == 0 && sub_count == 1) {
        acc_sub_row_from_avx2(d, s, sub_rows[0]);
        return true;
    }
    if (add_count == 1 && sub_count == 1) {
        acc_add_sub_row_from_avx2(d, s, add_rows[0], sub_rows[0]);
        return true;
    }
    // General case (castling / multi-piece batches): still a single fused pass —
    // read src once, fold every add/sub row, write dst once.
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + i));
        for (int j = 0; j < add_count; ++j) {
            a = _mm256_add_epi16(a, _mm256_load_si256(reinterpret_cast<const __m256i*>(add_rows[j] + i)));
        }
        for (int j = 0; j < sub_count; ++j) {
            a = _mm256_sub_epi16(a, _mm256_load_si256(reinterpret_cast<const __m256i*>(sub_rows[j] + i)));
        }
        _mm256_store_si256(reinterpret_cast<__m256i*>(d + i), a);
    }
    return true;
#elif defined(USE_NEON)
    if (add_count == 1 && sub_count == 0) {
        acc_add_row_from_neon(d, s, add_rows[0]);
        return true;
    }
    if (add_count == 0 && sub_count == 1) {
        acc_sub_row_from_neon(d, s, sub_rows[0]);
        return true;
    }
    if (add_count == 1 && sub_count == 1) {
        acc_add_sub_row_from_neon(d, s, add_rows[0], sub_rows[0]);
        return true;
    }
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 8) {
        int16x8_t a = vld1q_s16(s + i);
        for (int j = 0; j < add_count; ++j) {
            a = vaddq_s16(a, vld1q_s16(add_rows[j] + i));
        }
        for (int j = 0; j < sub_count; ++j) {
            a = vsubq_s16(a, vld1q_s16(sub_rows[j] + i));
        }
        vst1q_s16(d + i, a);
    }
    return true;
#else
    // Portable scalar fallback: materialize (if needed) then patch in place.
    if (d != s) {
        dst = src;
    }
    return update_accumulator_rows(dst, weights, updates, update_count);
#endif
}

// In-place patch: accumulator += add rows, -= sub rows. Retained as the fallback
// kernel for the non-NEON path of update_accumulator_rows_from (hence
// maybe_unused on NEON, where the fused kernel is always used instead).
[[maybe_unused]] bool update_accumulator_rows(std::array<int16_t, kExpectedFtSize>& accumulator,
                             const AlignedFtWeights& weights,
                             const AccumulatorFeatureUpdate* updates,
                             int update_count) {
    if (update_count <= 0) {
        return true;
    }
    if (updates == nullptr || update_count > kMaxBatchedPieceUpdates) {
        return false;
    }

    const int16_t* add_rows[kMaxBatchedPieceUpdates];
    const int16_t* sub_rows[kMaxBatchedPieceUpdates];
    int add_count = 0;
    int sub_count = 0;

    for (int i = 0; i < update_count; ++i) {
        const int feature = updates[i].feature;
        if (feature < 0) {
            return false;
        }

        const std::size_t offset = static_cast<std::size_t>(feature) * kExpectedFtSize;
        if (offset + kExpectedFtSize > weights.size()) {
            return false;
        }

        if (updates[i].op == AccumulatorUpdateOp::Add) {
            add_rows[add_count++] = weights.data() + offset;
        } else {
            sub_rows[sub_count++] = weights.data() + offset;
        }
    }

    int16_t* acc = accumulator.data();
#if defined(USE_AVX2)
    if (add_count == 1 && sub_count == 0) {
        acc_add_row_avx2(acc, add_rows[0]);
        return true;
    }
    if (add_count == 0 && sub_count == 1) {
        acc_sub_row_avx2(acc, sub_rows[0]);
        return true;
    }
    if (add_count == 1 && sub_count == 1) {
        acc_add_sub_row_avx2(acc, add_rows[0], sub_rows[0]);
        return true;
    }
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        for (int j = 0; j < add_count; ++j) {
            const __m256i r = _mm256_load_si256(reinterpret_cast<const __m256i*>(add_rows[j] + i));
            a = _mm256_add_epi16(a, r);
        }
        for (int j = 0; j < sub_count; ++j) {
            const __m256i r = _mm256_load_si256(reinterpret_cast<const __m256i*>(sub_rows[j] + i));
            a = _mm256_sub_epi16(a, r);
        }
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), a);
    }
#elif defined(USE_NEON)
    if (add_count == 1 && sub_count == 0) {
        acc_add_row_neon(acc, add_rows[0]);
        return true;
    }
    if (add_count == 0 && sub_count == 1) {
        acc_sub_row_neon(acc, sub_rows[0]);
        return true;
    }
    if (add_count == 1 && sub_count == 1) {
        acc_add_sub_row_neon(acc, add_rows[0], sub_rows[0]);
        return true;
    }
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 8) {
        int16x8_t a = vld1q_s16(acc + i);
        for (int j = 0; j < add_count; ++j) {
            a = vaddq_s16(a, vld1q_s16(add_rows[j] + i));
        }
        for (int j = 0; j < sub_count; ++j) {
            a = vsubq_s16(a, vld1q_s16(sub_rows[j] + i));
        }
        vst1q_s16(acc + i, a);
    }
#else
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        int32_t value = acc[i];
        for (int j = 0; j < add_count; ++j) {
            value += add_rows[j][i];
        }
        for (int j = 0; j < sub_count; ++j) {
            value -= sub_rows[j][i];
        }
        acc[i] = static_cast<int16_t>(value);
    }
#endif

    return true;
}

bool add_accumulator_row(std::array<int16_t, kExpectedFtSize>& accumulator,
                         const AlignedFtWeights& weights,
                         int feature) {
    return update_accumulator_row(accumulator, weights, feature, AccumulatorUpdateOp::Add);
}

bool subtract_accumulator_row(std::array<int16_t, kExpectedFtSize>& accumulator,
                              const AlignedFtWeights& weights,
                              int feature) {
    return update_accumulator_row(accumulator, weights, feature, AccumulatorUpdateOp::Subtract);
}

bool track_piece(thrawn::NnueState& state, int piece, int square) {
    if (piece < P || piece > k ||
        square < 0 || square >= BOARD_SIZE ||
        state.piece_count >= thrawn::NNUE_MAX_PIECES) {
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
    mark_accumulator_local(state, white_perspective);
    acc = network.ft_bias;

    for (int i = 0; i < static_cast<int>(state.piece_count); ++i) {
        const int piece = state.piece_list[i];
        const int square = state.square_list[i];
        const int feature = halfkav2_hm_index(piece, square, our_king_square, white_perspective);
        if (feature < 0 || !add_accumulator_row(acc, network.ft_weight, feature)) {
            return false;
        }
    }

    return true;
}

bool patch_piece_for_available_perspectives(thrawn::Position* pos,
                                            int ply,
                                            thrawn::NnueState& state,
                                            const LoadedNetwork& network,
                                            int piece,
                                            int square,
                                            AccumulatorUpdateOp op) {
    bool patched_any = false;

    if (state.white_king_sq >= 0 &&
        state.white_king_sq < BOARD_SIZE &&
        should_patch_piece_for_perspective(piece, true)) {
        if (!materialize_accumulator(pos, ply, true)) {
            return false;
        }
        const int feature = halfkav2_hm_index(piece, square, state.white_king_sq, true);
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

    if (state.black_king_sq >= 0 &&
        state.black_king_sq < BOARD_SIZE &&
        should_patch_piece_for_perspective(piece, false)) {
        if (!materialize_accumulator(pos, ply, false)) {
            return false;
        }
        const int feature = halfkav2_hm_index(piece, square, state.black_king_sq, false);
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

    return patched_any || is_king_piece(piece);
}

bool refresh_perspective_accumulator_from_board(const thrawn::Position* pos,
                                                thrawn::NnueState& state,
                                                const LoadedNetwork& network,
                                                bool white_perspective) {
    if (pos == nullptr) {
        return false;
    }

    auto& acc = white_perspective ? state.white_acc : state.black_acc;
    mark_accumulator_local(state, white_perspective);
    acc = network.ft_bias;

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
        uint64_t bitboard = pos->piece_bitboards[piece];
        while (bitboard) {
            const int square = pop_lsb(bitboard);
            const int feature = halfkav2_hm_index(piece, square, our_king_square, white_perspective);
            if (feature < 0 || !add_accumulator_row(acc, network.ft_weight, feature)) {
                return false;
            }
        }
    }

    return true;
}

bool add_piece_to_state(thrawn::Position* pos,
                        int ply,
                        thrawn::NnueState& state,
                        const LoadedNetwork& network,
                        int piece,
                        int square) {
    if (!patch_piece_for_available_perspectives(pos, ply, state, network, piece, square, AccumulatorUpdateOp::Add)) {
        return false;
    }

    return track_piece(state, piece, square);
}

bool remove_piece_from_state(thrawn::Position* pos,
                             int ply,
                             thrawn::NnueState& state,
                             const LoadedNetwork& network,
                             int piece,
                             int square) {
    if (!patch_piece_for_available_perspectives(pos, ply, state, network, piece, square, AccumulatorUpdateOp::Subtract)) {
        return false;
    }

    return untrack_piece(state, piece, square);
}

bool collect_feature_update_for_perspective(const thrawn::NnueState& state,
                                            const NnuePieceUpdate& update,
                                            bool white_perspective,
                                            AccumulatorFeatureUpdate* feature_updates,
                                            int& feature_update_count) {
    if (!should_patch_piece_for_perspective(update.piece, white_perspective)) {
        return true;
    }

    const int our_king_square = white_perspective ? state.white_king_sq : state.black_king_sq;
    if (our_king_square < 0 || our_king_square >= BOARD_SIZE) {
        return is_king_piece(update.piece);
    }

    const int feature = halfkav2_hm_index(update.piece,
                                          update.square,
                                          our_king_square,
                                          white_perspective);
    if (feature < 0) {
        return false;
    }
    if (feature_update_count >= kMaxBatchedPieceUpdates) {
        return false;
    }

    feature_updates[feature_update_count++] = {
        feature,
        update.add ? AccumulatorUpdateOp::Add : AccumulatorUpdateOp::Subtract
    };
    return true;
}

// Resolve the source accumulator for this perspective (the nearest materialized
// ancestor, or the local one) and write state's accumulator = source ± feature
// rows in a single fused pass, then mark it local. This replaces the previous
// two-step materialize (2 KB copy) + in-place patch with one read/write pass.
bool materialize_and_apply_rows(thrawn::Position* pos,
                                int ply,
                                thrawn::NnueState& state,
                                bool white_perspective,
                                const AlignedFtWeights& weights,
                                const AccumulatorFeatureUpdate* updates,
                                int update_count) {
    auto& dst = white_perspective ? state.white_acc : state.black_acc;
    int8_t& source = white_perspective ? state.white_acc_source_ply
                                       : state.black_acc_source_ply;

    const std::array<int16_t, kExpectedFtSize>* src_acc = &dst;
    if (source != kLocalAccumulatorSource) {
        if (!resolve_accumulator(pos, ply, white_perspective, src_acc) ||
            src_acc == nullptr) {
            return false;
        }
    }

    if (!update_accumulator_rows_from(dst, *src_acc, weights, updates, update_count)) {
        return false;
    }

    source = kLocalAccumulatorSource;
    return true;
}

bool apply_piece_updates_to_state(thrawn::Position* pos,
                                  int ply,
                                  thrawn::NnueState& state,
                                  const LoadedNetwork& network,
                                  const NnuePieceUpdate* updates,
                                  int update_count) {
    if (updates == nullptr || update_count < 0 || update_count > kMaxBatchedPieceUpdates) {
        return false;
    }

    bool refresh_white = false;
    bool refresh_black = false;
    for (int i = 0; i < update_count; ++i) {
        if (updates[i].piece == K) {
            refresh_white = true;
        } else if (updates[i].piece == k) {
            refresh_black = true;
        }
    }

    AccumulatorFeatureUpdate white_updates[kMaxBatchedPieceUpdates];
    AccumulatorFeatureUpdate black_updates[kMaxBatchedPieceUpdates];
    int white_update_count = 0;
    int black_update_count = 0;
    for (int i = 0; i < update_count; ++i) {
        if (!refresh_white &&
            !collect_feature_update_for_perspective(state, updates[i], true, white_updates, white_update_count)) {
            return false;
        }
        if (!refresh_black &&
            !collect_feature_update_for_perspective(state, updates[i], false, black_updates, black_update_count)) {
            return false;
        }
    }

    if (white_update_count > 0) {
        if (!materialize_and_apply_rows(pos, ply, state, true, network.ft_weight,
                                        white_updates, white_update_count)) {
            return false;
        }
    }
    if (black_update_count > 0) {
        if (!materialize_and_apply_rows(pos, ply, state, false, network.ft_weight,
                                        black_updates, black_update_count)) {
            return false;
        }
    }

    for (int i = 0; i < update_count; ++i) {
        const bool ok = updates[i].add
            ? track_piece(state, updates[i].piece, updates[i].square)
            : untrack_piece(state, updates[i].piece, updates[i].square);
        if (!ok) {
            return false;
        }
    }

    if (refresh_white && !refresh_perspective_accumulator_from_board(pos, state, network, true)) {
        return false;
    }
    if (refresh_black && !refresh_perspective_accumulator_from_board(pos, state, network, false)) {
        return false;
    }

    return true;
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

    if (!refresh_perspective_accumulator(out_state, network, true) ||
        !refresh_perspective_accumulator(out_state, network, false)) {
        clear_state(out_state);
        return false;
    }

    out_state.valid = true;
    return true;
}

EvaluationState state_for_evaluation(const thrawn::Position* pos,
                                     const LoadedNetwork& network,
                                     thrawn::NnueState& refreshed_state) {
    if (pos->ply >= 0 && pos->ply <= MAX_DEPTH) {
        const thrawn::NnueState& live_state = pos->nnue_stack[pos->ply];
        if (live_state.valid) {
            EvaluationState evaluation{&live_state, {}};
            if (accumulator_view_for_ply(pos, pos->ply, evaluation.accumulators)) {
                return evaluation;
            }
        }
    }

    if (!build_state_from_board(pos, network, refreshed_state)) {
        return {};
    }
    return {&refreshed_state, local_accumulator_view(refreshed_state)};
}

#if defined(USE_AVX2)
static inline __m256i add_dot_u8_i8_epi32(__m256i acc,
                                          __m256i x,
                                          __m256i w,
                                          const __m256i ones) {
#if defined(__AVXVNNI__)
    (void)ones;
    return _mm256_dpbusd_epi32(acc, x, w);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones));
#endif
}

static inline int32_t reduce_add_epi32(__m256i acc) {
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);

    int32_t sum = 0;
    for (int lane : lanes) {
        sum += lane;
    }
    return sum;
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

        acc0 = add_dot_u8_i8_epi32(acc0, xv0, wv0, ones);
        acc1 = add_dot_u8_i8_epi32(acc1, xv1, wv1, ones);
        acc2 = add_dot_u8_i8_epi32(acc2, xv2, wv2, ones);
        acc3 = add_dot_u8_i8_epi32(acc3, xv3, wv3, ones);
    }

    acc0 = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));

    for (; i < count; i += 32) {
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        acc0 = add_dot_u8_i8_epi32(acc0, xv, wv, ones);
    }

    return reduce_add_epi32(acc0);
}

void dot_u8_i8_x4_avx2(const uint8_t* x,
                       const int8_t* w0,
                       const int8_t* w1,
                       const int8_t* w2,
                       const int8_t* w3,
                       int count,
                       int32_t& s0,
                       int32_t& s1,
                       int32_t& s2,
                       int32_t& s3) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i a00 = _mm256_setzero_si256();
    __m256i a01 = _mm256_setzero_si256();
    __m256i a10 = _mm256_setzero_si256();
    __m256i a11 = _mm256_setzero_si256();
    __m256i a20 = _mm256_setzero_si256();
    __m256i a21 = _mm256_setzero_si256();
    __m256i a30 = _mm256_setzero_si256();
    __m256i a31 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 63 < count; i += 64) {
        const __m256i xv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        const __m256i xv1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + 32));

        a00 = add_dot_u8_i8_epi32(a00, xv0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w0 + i)), ones);
        a01 = add_dot_u8_i8_epi32(a01, xv1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w0 + i + 32)), ones);
        a10 = add_dot_u8_i8_epi32(a10, xv0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w1 + i)), ones);
        a11 = add_dot_u8_i8_epi32(a11, xv1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w1 + i + 32)), ones);
        a20 = add_dot_u8_i8_epi32(a20, xv0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w2 + i)), ones);
        a21 = add_dot_u8_i8_epi32(a21, xv1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w2 + i + 32)), ones);
        a30 = add_dot_u8_i8_epi32(a30, xv0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w3 + i)), ones);
        a31 = add_dot_u8_i8_epi32(a31, xv1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w3 + i + 32)), ones);
    }

    s0 = reduce_add_epi32(_mm256_add_epi32(a00, a01));
    s1 = reduce_add_epi32(_mm256_add_epi32(a10, a11));
    s2 = reduce_add_epi32(_mm256_add_epi32(a20, a21));
    s3 = reduce_add_epi32(_mm256_add_epi32(a30, a31));
}
#endif

#if defined(USE_NEON)
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
        return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi)));
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
        acc0 = accumulate(acc0, vld1q_s8(x + i), vld1q_s8(w + i));
    }
    return vaddvq_s32(acc0);
#endif
}

void dot_u8_i8_x4_neon(const uint8_t* x_u8,
                       const int8_t* w0,
                       const int8_t* w1,
                       const int8_t* w2,
                       const int8_t* w3,
                       int count,
                       int32_t& s0,
                       int32_t& s1,
                       int32_t& s2,
                       int32_t& s3) {
    const int8_t* x = reinterpret_cast<const int8_t*>(x_u8);

    int32x4_t a00 = vdupq_n_s32(0);
    int32x4_t a01 = vdupq_n_s32(0);
    int32x4_t a10 = vdupq_n_s32(0);
    int32x4_t a11 = vdupq_n_s32(0);
    int32x4_t a20 = vdupq_n_s32(0);
    int32x4_t a21 = vdupq_n_s32(0);
    int32x4_t a30 = vdupq_n_s32(0);
    int32x4_t a31 = vdupq_n_s32(0);

#if !defined(__ARM_FEATURE_DOTPROD)
    auto accumulate = [](int32x4_t acc, int8x16_t xv, int8x16_t wv) {
        const int16x8_t lo = vmull_s8(vget_low_s8(xv), vget_low_s8(wv));
        const int16x8_t hi = vmull_s8(vget_high_s8(xv), vget_high_s8(wv));
        return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi)));
    };
#endif

    int i = 0;
    for (; i + 31 < count; i += 32) {
        const int8x16_t x0 = vld1q_s8(x + i);
        const int8x16_t x1 = vld1q_s8(x + i + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        a00 = vdotq_s32(a00, x0, vld1q_s8(w0 + i));
        a01 = vdotq_s32(a01, x1, vld1q_s8(w0 + i + 16));
        a10 = vdotq_s32(a10, x0, vld1q_s8(w1 + i));
        a11 = vdotq_s32(a11, x1, vld1q_s8(w1 + i + 16));
        a20 = vdotq_s32(a20, x0, vld1q_s8(w2 + i));
        a21 = vdotq_s32(a21, x1, vld1q_s8(w2 + i + 16));
        a30 = vdotq_s32(a30, x0, vld1q_s8(w3 + i));
        a31 = vdotq_s32(a31, x1, vld1q_s8(w3 + i + 16));
#else
        a00 = accumulate(a00, x0, vld1q_s8(w0 + i));
        a01 = accumulate(a01, x1, vld1q_s8(w0 + i + 16));
        a10 = accumulate(a10, x0, vld1q_s8(w1 + i));
        a11 = accumulate(a11, x1, vld1q_s8(w1 + i + 16));
        a20 = accumulate(a20, x0, vld1q_s8(w2 + i));
        a21 = accumulate(a21, x1, vld1q_s8(w2 + i + 16));
        a30 = accumulate(a30, x0, vld1q_s8(w3 + i));
        a31 = accumulate(a31, x1, vld1q_s8(w3 + i + 16));
#endif
    }

    s0 = vaddvq_s32(vaddq_s32(a00, a01));
    s1 = vaddvq_s32(vaddq_s32(a10, a11));
    s2 = vaddvq_s32(vaddq_s32(a20, a21));
    s3 = vaddvq_s32(vaddq_s32(a30, a31));
}
#endif

void ft_pairwise_screlu_scalar(const int16_t* acc, uint8_t* out, int half, int ft_one, int shift) {
    for (int i = 0; i < half; ++i) {
        const int a = std::clamp<int>(acc[i], 0, ft_one);
        const int b = std::clamp<int>(acc[i + half], 0, ft_one);
        out[i] = static_cast<uint8_t>((a * b) >> shift);
    }
}

// Realize the FT activation (pairwise SqrCReLU) for one perspective into `out`,
// producing `half` uint8 activations. The SIMD paths are bit-identical to the
// scalar reference so incremental/refresh parity holds on every target.
void ft_pairwise_screlu(const int16_t* acc, uint8_t* out, int half, int ft_one, int shift) {
#if defined(USE_AVX2)
    if (half % 32 == 0) {
        ft_pairwise_screlu_avx2(acc, out, half, ft_one, shift);
        return;
    }
#elif defined(USE_NEON)
    if (half % 16 == 0) {
        ft_pairwise_screlu_neon(acc, out, half, ft_one, shift);
        return;
    }
#endif

    ft_pairwise_screlu_scalar(acc, out, half, ft_one, shift);
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

void dot_u8_i8_x4_scalar(const uint8_t* x,
                         const int8_t* w0,
                         const int8_t* w1,
                         const int8_t* w2,
                         const int8_t* w3,
                         int count,
                         int32_t& s0,
                         int32_t& s1,
                         int32_t& s2,
                         int32_t& s3) {
    int32_t a0 = 0;
    int32_t a1 = 0;
    int32_t a2 = 0;
    int32_t a3 = 0;
    for (int i = 0; i < count; ++i) {
        const int32_t xi = x[i];
        a0 += xi * static_cast<int32_t>(w0[i]);
        a1 += xi * static_cast<int32_t>(w1[i]);
        a2 += xi * static_cast<int32_t>(w2[i]);
        a3 += xi * static_cast<int32_t>(w3[i]);
    }
    s0 = a0;
    s1 = a1;
    s2 = a2;
    s3 = a3;
}

void dot_u8_i8_x4(const uint8_t* x,
                  const int8_t* w0,
                  const int8_t* w1,
                  const int8_t* w2,
                  const int8_t* w3,
                  int count,
                  int32_t& s0,
                  int32_t& s1,
                  int32_t& s2,
                  int32_t& s3) {
#if defined(USE_AVX2)
    if (count % 32 == 0) {
        dot_u8_i8_x4_avx2(x, w0, w1, w2, w3, count, s0, s1, s2, s3);
        return;
    }
#elif defined(USE_NEON)
    if (count % 16 == 0) {
        dot_u8_i8_x4_neon(x, w0, w1, w2, w3, count, s0, s1, s2, s3);
        return;
    }
#endif

    dot_u8_i8_x4_scalar(x, w0, w1, w2, w3, count, s0, s1, s2, s3);
}

uint8_t crelu_to_u8(int64_t value, int32_t scale) {
    if (value < 0) {
        return 0;
    }
    if (value > scale) {
        return static_cast<uint8_t>(scale);
    }
    return static_cast<uint8_t>(value);
}

uint8_t screlu_to_u8(uint8_t crelu, int32_t scale) {
    return static_cast<uint8_t>(div_round_by_scale(static_cast<int64_t>(crelu) * crelu, scale));
}

IntEvaluationResult evaluate_accumulators_int_quantized(const AccumulatorView& accumulators,
                                                        const LoadedNetwork& network,
                                                        int colour_to_move) {
    alignas(64) std::array<uint8_t, kFc0InputSize> act;
    alignas(64) std::array<int32_t, kExpectedFc0OutputSize> fc0;
    alignas(64) std::array<uint8_t, kFc1InputPaddedSize> a0;
    alignas(64) std::array<uint8_t, kExpectedFc1OutputSize> a1;

    const auto& us = (colour_to_move == white) ? *accumulators.white : *accumulators.black;
    const auto& them = (colour_to_move == white) ? *accumulators.black : *accumulators.white;

    // FT activation (pairwise SqrCReLU): [us_pairwise | them_pairwise], width FtSize.
    // The result carries an activation "one" of act_one, so the u8 x i8 fc0 dot is
    // renormalized by act_one to land the pre-bias sum back at the fc0 scale.
    ft_pairwise_screlu(us.data(), act.data(), kFtActivationHalf, network.ft_one, network.act_shift);
    ft_pairwise_screlu(them.data(), act.data() + kFtActivationHalf,
                       kFtActivationHalf, network.ft_one, network.act_shift);

    for (int j = 0; j < static_cast<int>(kExpectedFc0OutputSize); j += 4) {
        int32_t d0;
        int32_t d1;
        int32_t d2;
        int32_t d3;
        dot_u8_i8_x4(act.data(),
                     network.fc0_weight_t[j].data(),
                     network.fc0_weight_t[j + 1].data(),
                     network.fc0_weight_t[j + 2].data(),
                     network.fc0_weight_t[j + 3].data(),
                     kFc0InputSize,
                     d0, d1, d2, d3);

        fc0[j] = static_cast<int32_t>(static_cast<int64_t>(network.fc0_bias[j]) +
                                      div_round_by_scale(d0, network.act_one));
        fc0[j + 1] = static_cast<int32_t>(static_cast<int64_t>(network.fc0_bias[j + 1]) +
                                          div_round_by_scale(d1, network.act_one));
        fc0[j + 2] = static_cast<int32_t>(static_cast<int64_t>(network.fc0_bias[j + 2]) +
                                          div_round_by_scale(d2, network.act_one));
        fc0[j + 3] = static_cast<int32_t>(static_cast<int64_t>(network.fc0_bias[j + 3]) +
                                          div_round_by_scale(d3, network.act_one));
    }

    for (int i = 0; i < static_cast<int>(kExpectedHiddenSize); ++i) {
        const uint8_t crelu = crelu_to_u8(fc0[i], network.qfc0);
        a0[i] = screlu_to_u8(crelu, network.qfc0);
        a0[static_cast<int>(kExpectedHiddenSize) + i] = crelu;
    }
    if constexpr (kFc1InputPaddedSize > static_cast<int>(kExpectedFc1InputSize)) {
        std::fill(a0.begin() + static_cast<int>(kExpectedFc1InputSize), a0.end(), 0);
    }

    for (int j = 0; j < static_cast<int>(kExpectedFc1OutputSize); j += 4) {
        int32_t d0;
        int32_t d1;
        int32_t d2;
        int32_t d3;
        dot_u8_i8_x4(a0.data(),
                     network.fc1_weight_t[j].data(),
                     network.fc1_weight_t[j + 1].data(),
                     network.fc1_weight_t[j + 2].data(),
                     network.fc1_weight_t[j + 3].data(),
                     kFc1InputPaddedSize,
                     d0, d1, d2, d3);

        a1[j] = crelu_to_u8(static_cast<int64_t>(network.fc1_bias[j]) +
                            div_round_by_scale(d0, network.qfc0), network.qfc1);
        a1[j + 1] = crelu_to_u8(static_cast<int64_t>(network.fc1_bias[j + 1]) +
                                div_round_by_scale(d1, network.qfc0), network.qfc1);
        a1[j + 2] = crelu_to_u8(static_cast<int64_t>(network.fc1_bias[j + 2]) +
                                div_round_by_scale(d2, network.qfc0), network.qfc1);
        a1[j + 3] = crelu_to_u8(static_cast<int64_t>(network.fc1_bias[j + 3]) +
                                div_round_by_scale(d3, network.qfc0), network.qfc1);
    }

    const int32_t fc2_dot = dot_u8_i8(a1.data(), network.fc2_weight.data(), kExpectedFc1OutputSize);
    return {
        static_cast<int32_t>(static_cast<int64_t>(network.fc2_bias) +
                             div_round_by_scale(fc2_dot, network.qfc1)),
        fc0[static_cast<int>(kExpectedHiddenSize)]
    };
}

double raw_output_from_quantized(const IntEvaluationResult& result,
                                 const LoadedNetwork& network) {
    return static_cast<double>(result.fc2) * network.inv_fc2_scale +
           static_cast<double>(result.forward) * network.inv_fc0_scale;
}

int32_t evaluate_accumulators_int_score_stm(const AccumulatorView& accumulators,
                                            const LoadedNetwork& network,
                                            int colour_to_move) {
    const IntEvaluationResult result =
        evaluate_accumulators_int_quantized(accumulators, network, colour_to_move);
    return round_to_int32(raw_output_from_quantized(result, network) *
                          static_cast<double>(network.score_scale));
}

int32_t evaluate_accumulators_int_cp(const AccumulatorView& accumulators,
                                     const LoadedNetwork& network,
                                     int colour_to_move) {
    const IntEvaluationResult result =
        evaluate_accumulators_int_quantized(accumulators, network, colour_to_move);
    return round_to_int32(raw_output_from_quantized(result, network) *
                          static_cast<double>(network.score_scale) *
                          kCpPerStockfishScore);
}

int32_t evaluate_accumulators_int_engine_score(const AccumulatorView& accumulators,
                                               const LoadedNetwork& network,
                                               int colour_to_move) {
    if constexpr (kSearchUsesCentipawns) {
        return evaluate_accumulators_int_cp(accumulators, network, colour_to_move);
    } else {
        return evaluate_accumulators_int_score_stm(accumulators, network, colour_to_move);
    }
}

double crelu(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double evaluate_accumulators_float_raw_output(const AccumulatorView& accumulators,
                                             const LoadedNetwork& network,
                                             int colour_to_move) {
    std::array<double, kExpectedFtSize> act{};
    std::array<double, kExpectedFc0OutputSize> fc0{};
    std::array<double, kExpectedFc1InputSize> a0{};
    std::array<double, kExpectedFc1OutputSize> a1{};

    const auto& us = (colour_to_move == white) ? *accumulators.white : *accumulators.black;
    const auto& them = (colour_to_move == white) ? *accumulators.black : *accumulators.white;

    // FT activation (pairwise SqrCReLU): [us_pairwise | them_pairwise], width FtSize.
    for (int i = 0; i < kFtActivationHalf; ++i) {
        act[i] = crelu(static_cast<double>(us[i]) * network.inv_ft_scale) *
                 crelu(static_cast<double>(us[i + kFtActivationHalf]) * network.inv_ft_scale);
        act[kFtActivationHalf + i] =
            crelu(static_cast<double>(them[i]) * network.inv_ft_scale) *
            crelu(static_cast<double>(them[i + kFtActivationHalf]) * network.inv_ft_scale);
    }

    for (int j = 0; j < static_cast<int>(kExpectedFc0OutputSize); ++j) {
        double s = static_cast<double>(network.fc0_bias[j]) * network.inv_fc0_scale;
        const int8_t* weights = network.fc0_weight_t[j].data();
        for (int i = 0; i < static_cast<int>(kFc0InputSize); ++i) {
            s += act[i] * static_cast<double>(weights[i]) * network.inv_fc0_scale;
        }
        fc0[j] = s;
    }

    for (int i = 0; i < static_cast<int>(kExpectedHiddenSize); ++i) {
        const double c = crelu(fc0[i]);
        a0[i] = c * c;
        a0[static_cast<int>(kExpectedHiddenSize) + i] = c;
    }

    for (int j = 0; j < static_cast<int>(kExpectedFc1OutputSize); ++j) {
        double s = static_cast<double>(network.fc1_bias[j]) * network.inv_fc1_scale;
        for (int i = 0; i < static_cast<int>(kExpectedFc1InputSize); ++i) {
            s += a0[i] * static_cast<double>(network.fc1_weight_t[j][i]) * network.inv_fc1_scale;
        }
        a1[j] = crelu(s);
    }

    double out = static_cast<double>(network.fc2_bias) * network.inv_fc2_scale;
    for (int i = 0; i < static_cast<int>(kExpectedFc1OutputSize); ++i) {
        out += a1[i] * static_cast<double>(network.fc2_weight[i]) * network.inv_fc2_scale;
    }

    return out + fc0[static_cast<int>(kExpectedHiddenSize)];
}

double evaluate_accumulators_float_score_stm(const AccumulatorView& accumulators,
                                             const LoadedNetwork& network,
                                             int colour_to_move) {
    return evaluate_accumulators_float_raw_output(accumulators, network, colour_to_move) *
           static_cast<double>(network.score_scale);
}

double evaluate_accumulators_float_cp(const AccumulatorView& accumulators,
                                      const LoadedNetwork& network,
                                      int colour_to_move) {
    return evaluate_accumulators_float_score_stm(accumulators, network, colour_to_move) *
           kCpPerStockfishScore;
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

    AccumulatorView incremental_accumulators{};
    if (!accumulator_view_for_ply(pos, pos->ply, incremental_accumulators)) {
        return fail("incremental accumulator source invalid");
    }

    if (incremental.piece_count != rebuilt.piece_count) {
        return fail("piece count mismatch");
    }
    if (incremental.white_king_sq != rebuilt.white_king_sq ||
        incremental.black_king_sq != rebuilt.black_king_sq) {
        return fail("king square mismatch");
    }
    if (*incremental_accumulators.white != rebuilt.white_acc) {
        return fail("white accumulator mismatch");
    }
    if (*incremental_accumulators.black != rebuilt.black_acc) {
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

    const IntEvaluationResult incremental_raw =
        evaluate_accumulators_int_quantized(incremental_accumulators, network, pos->colour_to_move);
    const IntEvaluationResult rebuilt_raw =
        evaluate_accumulators_int_quantized(local_accumulator_view(rebuilt), network, pos->colour_to_move);
    if (!(incremental_raw == rebuilt_raw)) {
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
    const EvaluationState evaluation = state_for_evaluation(pos, network, refreshed_state);
    if (!evaluation.valid()) {
        return fail("failed to build evaluation state");
    }

    const int32_t int_cp = evaluate_accumulators_int_cp(evaluation.accumulators, network, pos->colour_to_move);
    const double float_cp = evaluate_accumulators_float_cp(evaluation.accumulators, network, pos->colour_to_move);
    if (abs_error_cp != nullptr) {
        *abs_error_cp = static_cast<float>(std::fabs(float_cp - static_cast<double>(int_cp)));
    }

    return true;
}

bool run_halfkav2_index_self_test(std::string& error) {
    for (const HalfKAv2SelfTestCase& test : kHalfKAv2SelfTests) {
        const int actual = halfkav2_hm_index(test.piece,
                                             test.piece_square,
                                             test.king_square,
                                             test.white_perspective);
        if (actual != test.expected_index) {
            error = "HalfKAv2_hm index self-test failed";
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
    uint32_t hidden_size = 0;
    uint32_t forward_size = 0;
    uint32_t fc0_output_size = 0;
    uint32_t fc0_input_size = 0;
    uint32_t fc1_input_size = 0;
    uint32_t fc1_output_size = 0;
    uint32_t output_perspective = 0;

    float ft_scale = 0.0f;
    float fc0_scale = 0.0f;
    float fc1_scale = 0.0f;
    float fc2_scale = 0.0f;
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
        !read_scalar_le(stream, hidden_size) ||
        !read_scalar_le(stream, forward_size) ||
        !read_scalar_le(stream, fc0_output_size) ||
        !read_scalar_le(stream, fc0_input_size) ||
        !read_scalar_le(stream, fc1_input_size) ||
        !read_scalar_le(stream, fc1_output_size) ||
        !read_scalar_le(stream, output_perspective) ||
        !read_scalar_le(stream, ft_scale) ||
        !read_scalar_le(stream, fc0_scale) ||
        !read_scalar_le(stream, fc1_scale) ||
        !read_scalar_le(stream, fc2_scale) ||
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
    if (num_features != kExpectedNumFeatures ||
        ft_size != kExpectedFtSize ||
        hidden_size != kExpectedHiddenSize ||
        forward_size != kExpectedForwardSize ||
        fc0_output_size != kExpectedFc0OutputSize ||
        fc0_input_size != kExpectedFc0InputSize ||
        fc1_input_size != kExpectedFc1InputSize ||
        fc1_output_size != kExpectedFc1OutputSize) {
        error = "unexpected network dimensions";
        return false;
    }
    // The pairwise SqrCReLU activation halves the fc0 input to the accumulator width.
    if (fc0_input_size != ft_size) {
        error = "fc0_input_size must equal ft_size";
        return false;
    }
    if (output_perspective != kExpectedOutputPerspective) {
        error = "unexpected output perspective";
        return false;
    }

    int32_t qft = 0;
    int32_t qfc0 = 0;
    int32_t qfc1 = 0;
    int32_t qfc2 = 0;

    if (!parse_integer_scale(ft_scale, "ft_scale", 255, qft, error) ||
        !parse_integer_scale(fc0_scale, "fc0_scale", 255, qfc0, error) ||
        !parse_integer_scale(fc1_scale, "fc1_scale", 255, qfc1, error) ||
        !parse_integer_scale(fc2_scale, "fc2_scale", 1 << 20, qfc2, error)) {
        return false;
    }
    if (!std::isfinite(score_scale) || score_scale <= 0.0f) {
        error = "invalid score_scale";
        return false;
    }

    // Integer realization of the pairwise SqrCReLU FT activation. Require ft_one+1
    // to be a power of two so the `>> act_shift` renormalization is exact, and
    // ft_one <= 255 so clamp(a)*clamp(b) < 2^16 (the SIMD 16-bit multiply relies on
    // this). act_shift = 2*log2(ft_one+1) - 7; act_one = (ft_one*ft_one) >> act_shift.
    const int32_t ft_one = qft;
    if (((ft_one + 1) & ft_one) != 0) {
        error = "ft_scale must be a power of two minus one";
        return false;
    }
    int32_t act_shift = -7;
    for (int32_t v = ft_one + 1; v > 1; v >>= 1) {
        act_shift += 2;
    }
    if (act_shift < 0) {
        error = "ft_scale too small for pairwise activation";
        return false;
    }
    const int32_t act_one = (ft_one * ft_one) >> act_shift;
    if (act_one <= 0 || act_one > 255) {
        error = "pairwise activation range invalid";
        return false;
    }
    if (description_length > (1U << 20)) {
        error = "description too large";
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
    AlignedFtWeights ft_weight_q;
    std::vector<int32_t> fc0_bias_q;
    std::vector<int8_t> fc0_weight_q;
    std::vector<int32_t> fc1_bias_q;
    std::vector<int8_t> fc1_weight_q;
    std::vector<int32_t> fc2_bias_q;
    std::vector<int8_t> fc2_weight_q;

    if (!read_vector_le(stream, ft_bias_q, kExpectedFtSize) ||
        !read_vector_le(stream, ft_weight_q, static_cast<std::size_t>(kExpectedNumFeatures) * kExpectedFtSize) ||
        !read_vector_le(stream, fc0_bias_q, kExpectedFc0OutputSize) ||
        !read_vector_le(stream, fc0_weight_q, static_cast<std::size_t>(kFc0InputSize) * kExpectedFc0OutputSize) ||
        !read_vector_le(stream, fc1_bias_q, kExpectedFc1OutputSize) ||
        !read_vector_le(stream, fc1_weight_q, static_cast<std::size_t>(kExpectedFc1InputSize) * kExpectedFc1OutputSize) ||
        !read_vector_le(stream, fc2_bias_q, kFc2OutputSize) ||
        !read_vector_le(stream, fc2_weight_q, kExpectedFc1OutputSize)) {
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
    network.fc0_scale = fc0_scale;
    network.fc1_scale = fc1_scale;
    network.fc2_scale = fc2_scale;
    network.score_scale = score_scale;

    network.qft = qft;
    network.qfc0 = qfc0;
    network.qfc1 = qfc1;
    network.qfc2 = qfc2;

    network.ft_one = ft_one;
    network.act_shift = act_shift;
    network.act_one = act_one;

    network.inv_ft_scale = 1.0f / ft_scale;
    network.inv_fc0_scale = 1.0f / fc0_scale;
    network.inv_fc1_scale = 1.0f / fc1_scale;
    network.inv_fc2_scale = 1.0f / fc2_scale;

    std::copy(ft_bias_q.begin(), ft_bias_q.end(), network.ft_bias.begin());
    network.ft_weight = std::move(ft_weight_q);
    std::copy(fc0_bias_q.begin(), fc0_bias_q.end(), network.fc0_bias.begin());
    std::copy(fc1_bias_q.begin(), fc1_bias_q.end(), network.fc1_bias.begin());

    for (int j = 0; j < static_cast<int>(kExpectedFc0OutputSize); ++j) {
        for (int i = 0; i < kFc0InputSize; ++i) {
            network.fc0_weight_t[j][i] =
                fc0_weight_q[static_cast<std::size_t>(i) * kExpectedFc0OutputSize + j];
        }
    }

    for (int j = 0; j < static_cast<int>(kExpectedFc1OutputSize); ++j) {
        for (int i = 0; i < static_cast<int>(kExpectedFc1InputSize); ++i) {
            network.fc1_weight_t[j][i] =
                fc1_weight_q[static_cast<std::size_t>(i) * kExpectedFc1OutputSize + j];
        }
    }

    network.fc2_bias = fc2_bias_q[0];
    std::copy(fc2_weight_q.begin(), fc2_weight_q.end(), network.fc2_weight.begin());

    network.description = std::move(description);
    network.loaded_path = path;

    if (!run_halfkav2_index_self_test(error)) {
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

    if (pos->ply >= 0 && pos->ply <= MAX_DEPTH) {
        const thrawn::NnueState& live_state = pos->nnue_stack[pos->ply];
        AccumulatorView accumulators{};
        if (live_state.valid && accumulator_view_for_ply(pos, pos->ply, accumulators)) {
            return evaluate_accumulators_int_engine_score(accumulators, *network, pos->colour_to_move);
        }
    }

    auto refreshed_state = std::make_unique<thrawn::NnueState>();
    if (!build_state_from_board(pos, *network, *refreshed_state)) {
        return 0;
    }

    return evaluate_accumulators_int_engine_score(local_accumulator_view(*refreshed_state),
                                                  *network,
                                                  pos->colour_to_move);
}

float nnue_evaluate_raw(const thrawn::Position* pos) {
    const LoadedNetwork* network = current_network();
    if (network == nullptr) {
        return 0.0f;
    }

    thrawn::NnueState refreshed_state;
    const EvaluationState evaluation = state_for_evaluation(pos, *network, refreshed_state);
    if (!evaluation.valid()) {
        return 0.0f;
    }

    return static_cast<float>(
        evaluate_accumulators_float_score_stm(evaluation.accumulators, *network, pos->colour_to_move));
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
#else
    (void)refreshed;
#endif
}

void nnue_copy_parent_to_child(thrawn::Position* pos, int child_ply) {
    if (child_ply <= 0 || child_ply > MAX_DEPTH) {
        return;
    }

    thrawn::NnueState& child = pos->nnue_stack[child_ply];
    const thrawn::NnueState& parent = pos->nnue_stack[child_ply - 1];
    if (current_network() == nullptr || !parent.valid) {
        child.valid = false;
        child.white_acc_source_ply = kLocalAccumulatorSource;
        child.black_acc_source_ply = kLocalAccumulatorSource;
        return;
    }

    copy_state_metadata_without_accumulators(child, parent, child_ply);
}

void nnue_promote_to_root(thrawn::Position* pos, int ply) {
    if (pos == nullptr || ply < 0 || ply > MAX_DEPTH) {
        return;
    }

    thrawn::NnueState& root = pos->nnue_stack[0];
    if (current_network() == nullptr) {
        root.valid = false;
        root.white_acc_source_ply = kLocalAccumulatorSource;
        root.black_acc_source_ply = kLocalAccumulatorSource;
        return;
    }

    if (!pos->nnue_stack[ply].valid || !materialize_state_accumulators(pos, ply)) {
        root.valid = false;
        root.white_acc_source_ply = kLocalAccumulatorSource;
        root.black_acc_source_ply = kLocalAccumulatorSource;
        return;
    }

    root = pos->nnue_stack[ply];
    root.white_acc_source_ply = kLocalAccumulatorSource;
    root.black_acc_source_ply = kLocalAccumulatorSource;
}

void nnue_apply_piece_updates(thrawn::Position* pos,
                              int ply,
                              const NnuePieceUpdate* updates,
                              int update_count) {
    const LoadedNetwork* network = current_network();
    if (network == nullptr ||
        pos == nullptr ||
        updates == nullptr ||
        update_count < 0 ||
        ply < 0 ||
        ply > MAX_DEPTH) {
        return;
    }

    thrawn::NnueState& state = pos->nnue_stack[ply];
    if (!state.valid) {
        return;
    }

    if (!apply_piece_updates_to_state(pos, ply, state, *network, updates, update_count)) {
        state.valid = false;
    }
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

    if (!add_piece_to_state(pos, ply, state, *network, piece, square)) {
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

    if (!remove_piece_from_state(pos, ply, state, *network, piece, square)) {
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
