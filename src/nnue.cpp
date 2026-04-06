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
constexpr uint32_t kCurrentVersion = 3;
constexpr uint32_t kExpectedNumFeatures = thrawn::NNUE_INPUT_FEATURES;
constexpr uint32_t kExpectedFtSize = thrawn::NNUE_ACCUMULATOR_SIZE;
constexpr uint32_t kExpectedHiddenSize = thrawn::NNUE_HIDDEN_SIZE;
constexpr uint32_t kExpectedOutputBuckets = thrawn::NNUE_OUTPUT_BUCKETS;
constexpr uint32_t kExpectedOutputPerspective = 1;
constexpr const char* kExpectedFeatureSet = "a768_dual_v1";

struct LoadedNetwork {
    uint32_t version = 0;
    uint32_t output_buckets = 0;
    float inv_ft_scale = 0.0f;
    std::array<int16_t, kExpectedFtSize> ft_bias{};
    std::vector<int16_t> ft_weight;
    std::array<float, kExpectedHiddenSize> l1_bias{};
    std::array<float, kExpectedFtSize * 2 * kExpectedHiddenSize> l1_weight{};
    std::array<float, kExpectedOutputBuckets> out_bias{};
    std::array<float, kExpectedHiddenSize * kExpectedOutputBuckets> out_weight{};
    std::string loaded_path;
};

std::mutex g_network_mutex;
std::shared_ptr<const LoadedNetwork> g_network_owner;
std::atomic<const LoadedNetwork*> g_network_raw{nullptr};

template <typename T>
bool read_scalar(std::ifstream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

template <typename T>
bool read_vector(std::ifstream& stream, std::vector<T>& values, std::size_t count) {
    values.resize(count);
    if (count == 0) {
        return true;
    }
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<bool>(stream);
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
        path = "model.nnue";
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

int engine_to_model_square(int square) {
    return square ^ 56;
}

int flip_vertical(int square) {
    return ((7 - (square / 8)) * 8) + (square % 8);
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
        case K:
        case k:
            return 5;
        default:
            return -1;
    }
}

bool is_white_piece(int piece) {
    return piece >= P && piece <= K;
}

int relative_color_bit(int piece, bool white_perspective) {
    return white_perspective
        ? (is_white_piece(piece) ? 0 : 1)
        : (is_white_piece(piece) ? 1 : 0);
}

int oriented_square(int engine_square, bool white_perspective) {
    const int white_square = engine_to_model_square(engine_square);
    return white_perspective ? white_square : flip_vertical(white_square);
}

int feature_index(int piece, int engine_square, bool white_perspective) {
    const int piece_index = piece_type_index(piece);
    if (piece_index < 0 || engine_square < 0 || engine_square >= BOARD_SIZE) {
        return -1;
    }

    return (piece_index * 2 + relative_color_bit(piece, white_perspective)) * 64 +
           oriented_square(engine_square, white_perspective);
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
    state.valid = false;
}

float clip_unit(float value) {
    if (value <= 0.0f) {
        return 0.0f;
    }
    if (value >= 1.0f) {
        return 1.0f;
    }
    return value;
}

int finalize_evaluation(float output) {
    if (!std::isfinite(output)) {
        return 0;
    }

    if (output > static_cast<float>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    if (output < static_cast<float>(std::numeric_limits<int>::min())) {
        return std::numeric_limits<int>::min();
    }

    return static_cast<int>(std::lround(output));
}

int output_bucket_index(int piece_count) {
    constexpr int kMinPieceCount = 2;
    constexpr int kMaxPieceCount = 32;
    constexpr int kPhaseSpan = kMaxPieceCount - kMinPieceCount + 1;

    const int clamped_piece_count = std::min(kMaxPieceCount, std::max(kMinPieceCount, piece_count));
    const int phase_progress = kMaxPieceCount - clamped_piece_count;
    return std::min(
        static_cast<int>(kExpectedOutputBuckets) - 1,
        (phase_progress * static_cast<int>(kExpectedOutputBuckets)) / kPhaseSpan
    );
}

enum class AccumulatorUpdateOp { Add, Subtract };

#if defined(USE_AVX2)
using AccumulatorVec = __m256i;
constexpr int kAccumulatorSimdWidth = 16;
static_assert(kExpectedFtSize % kAccumulatorSimdWidth == 0, "AVX2 accumulator width mismatch");

inline AccumulatorVec accumulator_load(const int16_t* data) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
}

inline void accumulator_store(int16_t* data, AccumulatorVec value) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(data), value);
}

inline AccumulatorVec accumulator_add(AccumulatorVec lhs, AccumulatorVec rhs) {
    return _mm256_add_epi16(lhs, rhs);
}

inline AccumulatorVec accumulator_subtract(AccumulatorVec lhs, AccumulatorVec rhs) {
    return _mm256_sub_epi16(lhs, rhs);
}
#elif defined(USE_NEON)
using AccumulatorVec = int16x8_t;
constexpr int kAccumulatorSimdWidth = 8;
static_assert(kExpectedFtSize % kAccumulatorSimdWidth == 0, "NEON accumulator width mismatch");

inline AccumulatorVec accumulator_load(const int16_t* data) {
    return vld1q_s16(data);
}

inline void accumulator_store(int16_t* data, AccumulatorVec value) {
    vst1q_s16(data, value);
}

inline AccumulatorVec accumulator_add(AccumulatorVec lhs, AccumulatorVec rhs) {
    return vaddq_s16(lhs, rhs);
}

inline AccumulatorVec accumulator_subtract(AccumulatorVec lhs, AccumulatorVec rhs) {
    return vsubq_s16(lhs, rhs);
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

#if defined(USE_AVX2) || defined(USE_NEON)
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += kAccumulatorSimdWidth) {
        const AccumulatorVec acc_vec = accumulator_load(acc + i);
        const AccumulatorVec row_vec = accumulator_load(row + i);
        const AccumulatorVec out_vec = (op == AccumulatorUpdateOp::Add)
            ? accumulator_add(acc_vec, row_vec)
            : accumulator_subtract(acc_vec, row_vec);
        accumulator_store(acc + i, out_vec);
    }
#else
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        const int updated = (op == AccumulatorUpdateOp::Add)
            ? (accumulator[i] + row[i])
            : (accumulator[i] - row[i]);
        accumulator[i] = static_cast<int16_t>(updated);
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

void accumulate_hidden_row_scalar(std::array<float, kExpectedHiddenSize>& hidden,
                                  const float* row_weights,
                                  float value) {
    for (int j = 0; j < static_cast<int>(kExpectedHiddenSize); ++j) {
        hidden[j] += value * row_weights[j];
    }
}

#if defined(USE_AVX2)
void accumulate_hidden_row_avx2(std::array<float, kExpectedHiddenSize>& hidden,
                                const float* row_weights,
                                float value) {
    const __m256 value_vec = _mm256_set1_ps(value);
    float* hidden_data = hidden.data();

    for (int j = 0; j < static_cast<int>(kExpectedHiddenSize); j += 8) {
        const __m256 hidden_vec = _mm256_loadu_ps(hidden_data + j);
        const __m256 weight_vec = _mm256_loadu_ps(row_weights + j);
        const __m256 product_vec = _mm256_mul_ps(value_vec, weight_vec);
        const __m256 updated_vec = _mm256_add_ps(hidden_vec, product_vec);
        _mm256_storeu_ps(hidden_data + j, updated_vec);
    }
}
#endif

#if defined(USE_NEON)
void accumulate_hidden_row_neon(std::array<float, kExpectedHiddenSize>& hidden,
                                const float* row_weights,
                                float value) {
    const float32x4_t value_vec = vdupq_n_f32(value);
    float* hidden_data = hidden.data();

    for (int j = 0; j < static_cast<int>(kExpectedHiddenSize); j += 4) {
        const float32x4_t hidden_vec = vld1q_f32(hidden_data + j);
        const float32x4_t weight_vec = vld1q_f32(row_weights + j);
        const float32x4_t product_vec = vmulq_f32(value_vec, weight_vec);
        const float32x4_t updated_vec = vaddq_f32(hidden_vec, product_vec);
        vst1q_f32(hidden_data + j, updated_vec);
    }
}
#endif

void accumulate_hidden_row(std::array<float, kExpectedHiddenSize>& hidden,
                           const float* row_weights,
                           float value) {
#if defined(USE_AVX2)
    accumulate_hidden_row_avx2(hidden, row_weights, value);
#elif defined(USE_NEON)
    accumulate_hidden_row_neon(hidden, row_weights, value);
#else
    accumulate_hidden_row_scalar(hidden, row_weights, value);
#endif
}

float output_sum_scalar(const std::array<float, kExpectedHiddenSize>& hidden,
                        const LoadedNetwork& network,
                        int bucket) {
    float output = network.out_bias[bucket];
    for (int j = 0; j < static_cast<int>(kExpectedHiddenSize); ++j) {
        output += clip_unit(hidden[j]) * network.out_weight[j * kExpectedOutputBuckets + bucket];
    }
    return output;
}

bool add_piece_to_state(thrawn::NnueState& state, const LoadedNetwork& network, int piece, int square) {
    if (square < 0 || square >= BOARD_SIZE || state.piece_count >= thrawn::NNUE_MAX_PIECES) {
        return false;
    }

    if (state.index_by_square[square] != -1) {
        return false;
    }

    const int white_feature = feature_index(piece, square, true);
    const int black_feature = feature_index(piece, square, false);
    if (!add_accumulator_row(state.white_acc, network.ft_weight, white_feature) ||
        !add_accumulator_row(state.black_acc, network.ft_weight, black_feature)) {
        return false;
    }

    const int index = state.piece_count;
    state.piece_list[index] = static_cast<uint8_t>(piece);
    state.square_list[index] = static_cast<uint8_t>(square);
    state.index_by_square[square] = static_cast<int8_t>(index);
    state.piece_count = static_cast<uint8_t>(state.piece_count + 1);
    return true;
}

bool remove_piece_from_state(thrawn::NnueState& state, const LoadedNetwork& network, int piece, int square) {
    if (square < 0 || square >= BOARD_SIZE) {
        return false;
    }

    const int index = state.index_by_square[square];
    if (index < 0 || index >= state.piece_count || state.piece_list[index] != piece) {
        return false;
    }

    const int white_feature = feature_index(piece, square, true);
    const int black_feature = feature_index(piece, square, false);
    if (!subtract_accumulator_row(state.white_acc, network.ft_weight, white_feature) ||
        !subtract_accumulator_row(state.black_acc, network.ft_weight, black_feature)) {
        return false;
    }

    const int last_index = state.piece_count - 1;
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

bool build_state_from_board(const thrawn::Position* pos, const LoadedNetwork& network, thrawn::NnueState& out_state) {
    clear_state(out_state);
    out_state.white_acc = network.ft_bias;
    out_state.black_acc = network.ft_bias;

    for (int piece = P; piece <= k; ++piece) {
        uint64_t bitboard = pos->piece_bitboards[piece];
        while (bitboard) {
            const int square = get_lsb_index(bitboard);
            if (!add_piece_to_state(out_state, network, piece, square)) {
                clear_state(out_state);
                return false;
            }
            pop_bit(bitboard, square);
        }
    }

    out_state.valid = true;
    return true;
}

int evaluate_state_scalar(const thrawn::NnueState& state, const LoadedNetwork& network, int colour_to_move) {
    std::array<float, kExpectedHiddenSize> hidden = network.l1_bias;
    const int bucket = output_bucket_index(state.piece_count);

    const auto& stm_acc = (colour_to_move == white) ? state.white_acc : state.black_acc;
    const auto& nstm_acc = (colour_to_move == white) ? state.black_acc : state.white_acc;

    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        const float value = clip_unit(static_cast<float>(stm_acc[i]) * network.inv_ft_scale);
        if (value == 0.0f) {
            continue;
        }

        const std::size_t row = static_cast<std::size_t>(i) * kExpectedHiddenSize;
        accumulate_hidden_row_scalar(hidden, network.l1_weight.data() + row, value);
    }

    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        const float value = clip_unit(static_cast<float>(nstm_acc[i]) * network.inv_ft_scale);
        if (value == 0.0f) {
            continue;
        }

        const std::size_t row = static_cast<std::size_t>(kExpectedFtSize + i) * kExpectedHiddenSize;
        accumulate_hidden_row_scalar(hidden, network.l1_weight.data() + row, value);
    }

    return finalize_evaluation(output_sum_scalar(hidden, network, bucket));
}

int evaluate_state_simd(const thrawn::NnueState& state, const LoadedNetwork& network, int colour_to_move) {
    std::array<float, kExpectedHiddenSize> hidden = network.l1_bias;
    const int bucket = output_bucket_index(state.piece_count);

    const auto& stm_acc = (colour_to_move == white) ? state.white_acc : state.black_acc;
    const auto& nstm_acc = (colour_to_move == white) ? state.black_acc : state.white_acc;

    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        const float value = clip_unit(static_cast<float>(stm_acc[i]) * network.inv_ft_scale);
        if (value == 0.0f) {
            continue;
        }

        const std::size_t row = static_cast<std::size_t>(i) * kExpectedHiddenSize;
        accumulate_hidden_row(hidden, network.l1_weight.data() + row, value);
    }

    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        const float value = clip_unit(static_cast<float>(nstm_acc[i]) * network.inv_ft_scale);
        if (value == 0.0f) {
            continue;
        }

        const std::size_t row = static_cast<std::size_t>(kExpectedFtSize + i) * kExpectedHiddenSize;
        accumulate_hidden_row(hidden, network.l1_weight.data() + row, value);
    }

    return finalize_evaluation(output_sum_scalar(hidden, network, bucket));
}

int evaluate_state(const thrawn::NnueState& state, const LoadedNetwork& network, int colour_to_move) {
#if defined(USE_AVX2) || defined(USE_NEON)
    return evaluate_state_simd(state, network, colour_to_move);
#else
    return evaluate_state_scalar(state, network, colour_to_move);
#endif
}

bool load_network_from_file(const std::string& path, LoadedNetwork& network, std::string& error) {
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
    uint32_t output_buckets = 0;
    uint32_t output_perspective = 0;
    float ft_scale = 0.0f;
    float dense_scale = 0.0f;
    float wdl_scale = 0.0f;
    uint32_t description_length = 0;

    if (!read_scalar(stream, version)) {
        error = "failed while reading version";
        return false;
    }

    const std::string feature_set = read_feature_set(stream);
    if (!stream ||
        !read_scalar(stream, num_features) ||
        !read_scalar(stream, ft_size) ||
        !read_scalar(stream, hidden_size) ||
        !read_scalar(stream, output_buckets) ||
        !read_scalar(stream, output_perspective) ||
        !read_scalar(stream, ft_scale) ||
        !read_scalar(stream, dense_scale) ||
        !read_scalar(stream, wdl_scale) ||
        !read_scalar(stream, description_length)) {
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
    if (num_features != kExpectedNumFeatures) {
        error = "unexpected feature count";
        return false;
    }
    if (ft_size != kExpectedFtSize || hidden_size != kExpectedHiddenSize) {
        error = "unexpected network dimensions";
        return false;
    }
    if (output_buckets != kExpectedOutputBuckets) {
        error = "unexpected output bucket count";
        return false;
    }
    if (output_perspective != kExpectedOutputPerspective) {
        error = "unexpected output perspective";
        return false;
    }
    if (ft_scale <= 0.0f || dense_scale <= 0.0f) {
        error = "invalid quantization scales";
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

    if (!read_vector(stream, ft_bias_q, kExpectedFtSize) ||
        !read_vector(stream, ft_weight_q, static_cast<std::size_t>(kExpectedNumFeatures) * kExpectedFtSize) ||
        !read_vector(stream, l1_bias_q, kExpectedHiddenSize) ||
        !read_vector(stream, l1_weight_q, static_cast<std::size_t>(kExpectedFtSize) * 2 * kExpectedHiddenSize)) {
        error = "failed while reading tensors";
        return false;
    }

    network = LoadedNetwork{};
    network.version = version;
    network.output_buckets = output_buckets;
    network.inv_ft_scale = 1.0f / ft_scale;
    std::copy(ft_bias_q.begin(), ft_bias_q.end(), network.ft_bias.begin());
    network.ft_weight = std::move(ft_weight_q);

    for (int i = 0; i < static_cast<int>(kExpectedHiddenSize); ++i) {
        network.l1_bias[i] = static_cast<float>(l1_bias_q[i]) / dense_scale;
    }
    for (std::size_t i = 0; i < network.l1_weight.size(); ++i) {
        network.l1_weight[i] = static_cast<float>(l1_weight_q[i]) / dense_scale;
    }

    std::vector<int32_t> out_bias_q;
    std::vector<int16_t> out_weight_q;
    if (!read_vector(stream, out_bias_q, kExpectedOutputBuckets) ||
        !read_vector(stream, out_weight_q, static_cast<std::size_t>(kExpectedHiddenSize) * kExpectedOutputBuckets)) {
        error = "failed while reading output layer";
        return false;
    }

    for (int i = 0; i < static_cast<int>(kExpectedOutputBuckets); ++i) {
        network.out_bias[i] = static_cast<float>(out_bias_q[i]) / dense_scale;
    }
    for (std::size_t i = 0; i < network.out_weight.size(); ++i) {
        network.out_weight[i] = static_cast<float>(out_weight_q[i]) / dense_scale;
    }

    if (!stream) {
        error = "unexpected end of file";
        return false;
    }

    network.loaded_path = path;
    (void)wdl_scale;
    (void)description;
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

    const thrawn::NnueState& state = pos->nnue_stack[pos->ply];
    if (!state.valid) {
        return 0;
    }

    return evaluate_state(state, *network, pos->colour_to_move);
}

void nnue_refresh_root(thrawn::Position* pos) {
    thrawn::NnueState& root = pos->nnue_stack[0];
    const LoadedNetwork* network = current_network();
    if (network == nullptr) {
        clear_state(root);
        return;
    }

    build_state_from_board(pos, *network, root);
#ifdef DEBUG_BUILD
    nnue_debug_check(pos);
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

void nnue_debug_check(const thrawn::Position* pos) {
#ifndef DEBUG_BUILD
    (void)pos;
#else
    const LoadedNetwork* network = current_network();
    if (network == nullptr) {
        return;
    }

    if (pos->ply < 0 || pos->ply > MAX_DEPTH) {
        parity_failure("invalid ply", pos);
    }

    thrawn::NnueState rebuilt;
    if (!build_state_from_board(pos, *network, rebuilt)) {
        parity_failure("failed full refresh", pos);
    }

    const thrawn::NnueState& incremental = pos->nnue_stack[pos->ply];
    if (!incremental.valid) {
        parity_failure("incremental state invalid", pos);
    }

    if (incremental.piece_count != rebuilt.piece_count ||
        incremental.white_acc != rebuilt.white_acc ||
        incremental.black_acc != rebuilt.black_acc) {
        parity_failure("accumulator mismatch", pos);
    }

    for (int square = 0; square < BOARD_SIZE; ++square) {
        const int inc_index = incremental.index_by_square[square];
        const int ref_index = rebuilt.index_by_square[square];
        if ((inc_index == -1) != (ref_index == -1)) {
            parity_failure("piece presence mismatch", pos);
        }
        if (inc_index != -1 && incremental.piece_list[inc_index] != rebuilt.piece_list[ref_index]) {
            parity_failure("piece identity mismatch", pos);
        }
    }

    const int incremental_eval = evaluate_state(incremental, *network, pos->colour_to_move);
    const int rebuilt_eval = evaluate_state(rebuilt, *network, pos->colour_to_move);
    if (incremental_eval != rebuilt_eval) {
        parity_failure("evaluation mismatch", pos);
    }

#if defined(USE_AVX2) || defined(USE_NEON)
    if (incremental_eval != evaluate_state_scalar(incremental, *network, pos->colour_to_move) ||
        rebuilt_eval != evaluate_state_scalar(rebuilt, *network, pos->colour_to_move)) {
        parity_failure("simd scalar mismatch", pos);
    }
#endif
#endif
}
