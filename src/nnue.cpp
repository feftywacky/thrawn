#include "nnue.h"

#include "bitboard.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
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
constexpr int32_t kExpectedFtScale = 127;
constexpr int32_t kExpectedDenseScale = 96;
constexpr int32_t kNormalizationConstant = 50;
constexpr int32_t kRawScaleInt = kExpectedDenseScale * kExpectedDenseScale;
constexpr int64_t kCpScaleDenominator = static_cast<int64_t>(kRawScaleInt) * kNormalizationConstant;
constexpr float kRawScale = static_cast<float>(kRawScaleInt);

struct LoadedNetwork {
    uint32_t version = 0;
    uint32_t output_buckets = 0;
    std::array<int16_t, kExpectedFtSize> ft_bias{};
    std::vector<int16_t> ft_weight;
    std::array<int32_t, kExpectedHiddenSize> l1_bias{};
    std::vector<int8_t> l1_weight;
    std::array<int32_t, kExpectedOutputBuckets> out_bias{};
    std::vector<int16_t> out_weight;
    std::string loaded_path;
};

std::mutex g_network_mutex;
std::shared_ptr<const LoadedNetwork> g_network_owner;
std::atomic<const LoadedNetwork*> g_network_raw{nullptr};

template <typename T>
bool read_scalar_le(std::ifstream& stream, T& value) {
    std::array<unsigned char, sizeof(T)> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        return false;
    }

    const uint16_t endian_probe = 0x0102;
    const bool host_is_little_endian = *reinterpret_cast<const unsigned char*>(&endian_probe) == 0x02;
    if (host_is_little_endian) {
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

void copy_ft_bias(std::array<int16_t, kExpectedFtSize>& accumulator,
                  const std::array<int16_t, kExpectedFtSize>& bias) {
    accumulator = bias;
}

int output_bucket_index(int piece_count, int output_buckets) {
    constexpr int kMinPieceCount = 2;
    constexpr int kMaxPieceCount = 32;

    if (output_buckets <= 1) {
        return 0;
    }

    const int clamped_piece_count = std::min(kMaxPieceCount, std::max(kMinPieceCount, piece_count));
    const int phase_progress = kMaxPieceCount - clamped_piece_count;
    return std::min(output_buckets - 1, (phase_progress * output_buckets) / 31);
}

enum class AccumulatorUpdateOp { Add, Subtract };

#if defined(USE_AVX2)
void update_accumulator_row_avx2(std::array<int16_t, kExpectedFtSize>& accumulator,
                                 const int16_t* row,
                                 AccumulatorUpdateOp op) {
    int16_t* acc = accumulator.data();
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 16) {
        __m256i acc_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i row_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
        acc_vec = (op == AccumulatorUpdateOp::Add)
            ? _mm256_add_epi16(acc_vec, row_vec)
            : _mm256_sub_epi16(acc_vec, row_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), acc_vec);
    }
}
#endif

#if defined(USE_NEON)
void update_accumulator_row_neon(std::array<int16_t, kExpectedFtSize>& accumulator,
                                 const int16_t* row,
                                 AccumulatorUpdateOp op) {
    int16_t* acc = accumulator.data();
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); i += 8) {
        int16x8_t acc_vec = vld1q_s16(acc + i);
        const int16x8_t row_vec = vld1q_s16(row + i);
        acc_vec = (op == AccumulatorUpdateOp::Add)
            ? vaddq_s16(acc_vec, row_vec)
            : vsubq_s16(acc_vec, row_vec);
        vst1q_s16(acc + i, acc_vec);
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
#if defined(USE_AVX2)
    update_accumulator_row_avx2(accumulator, row, op);
#elif defined(USE_NEON)
    update_accumulator_row_neon(accumulator, row, op);
#else
    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        accumulator[i] = (op == AccumulatorUpdateOp::Add)
            ? static_cast<int16_t>(accumulator[i] + row[i])
            : static_cast<int16_t>(accumulator[i] - row[i]);
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
    if (index < 0 || index >= static_cast<int>(state.piece_count) || state.piece_list[index] != piece) {
        return false;
    }

    const int white_feature = feature_index(piece, square, true);
    const int black_feature = feature_index(piece, square, false);
    if (!subtract_accumulator_row(state.white_acc, network.ft_weight, white_feature) ||
        !subtract_accumulator_row(state.black_acc, network.ft_weight, black_feature)) {
        return false;
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

bool build_state_from_board(const thrawn::Position* pos, const LoadedNetwork& network, thrawn::NnueState& out_state) {
    clear_state(out_state);
    copy_ft_bias(out_state.white_acc, network.ft_bias);
    copy_ft_bias(out_state.black_acc, network.ft_bias);

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

int32_t evaluate_state_raw_qb2(const thrawn::NnueState& state, const LoadedNetwork& network, int colour_to_move) {
    std::array<int32_t, kExpectedFtSize * 2> screlu{};
    std::array<int32_t, kExpectedHiddenSize> hidden{};

    const auto& us = (colour_to_move == white) ? state.white_acc : state.black_acc;
    const auto& them = (colour_to_move == white) ? state.black_acc : state.white_acc;

    for (int i = 0; i < static_cast<int>(kExpectedFtSize); ++i) {
        const int32_t us_clamped = std::clamp(static_cast<int32_t>(us[i]), 0, kExpectedFtScale);
        const int32_t them_clamped = std::clamp(static_cast<int32_t>(them[i]), 0, kExpectedFtScale);
        screlu[i] = us_clamped * us_clamped;
        screlu[kExpectedFtSize + i] = them_clamped * them_clamped;
    }

    constexpr int64_t ft_scale_sq = static_cast<int64_t>(kExpectedFtScale) * kExpectedFtScale;
    for (int j = 0; j < static_cast<int>(kExpectedHiddenSize); ++j) {
        int64_t sum = static_cast<int64_t>(network.l1_bias[j]) * ft_scale_sq;
        for (int i = 0; i < static_cast<int>(kExpectedFtSize) * 2; ++i) {
            sum += static_cast<int64_t>(screlu[i]) *
                   static_cast<int64_t>(network.l1_weight[static_cast<std::size_t>(i) * kExpectedHiddenSize + j]);
        }
        sum /= ft_scale_sq;
        hidden[j] = static_cast<int32_t>(std::clamp<int64_t>(sum, 0, kExpectedDenseScale));
    }

    const int bucket = output_bucket_index(state.piece_count, static_cast<int>(network.output_buckets));
    int64_t output = 0;
    for (int j = 0; j < static_cast<int>(kExpectedHiddenSize); ++j) {
        output += static_cast<int64_t>(hidden[j]) *
                  static_cast<int64_t>(network.out_weight[static_cast<std::size_t>(j) * network.output_buckets + bucket]);
    }
    output += static_cast<int64_t>(network.out_bias[bucket]) * kExpectedDenseScale;

    if (output > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    if (output < static_cast<int64_t>(std::numeric_limits<int32_t>::min())) {
        return std::numeric_limits<int32_t>::min();
    }
    return static_cast<int32_t>(output);
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

    const int32_t incremental_raw = evaluate_state_raw_qb2(incremental, network, pos->colour_to_move);
    const int32_t rebuilt_raw = evaluate_state_raw_qb2(rebuilt, network, pos->colour_to_move);
    if (incremental_raw != rebuilt_raw) {
        return fail("raw evaluation mismatch");
    }

    return true;
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

    if (!read_scalar_le(stream, version)) {
        error = "failed while reading version";
        return false;
    }

    const std::string feature_set = read_feature_set(stream);
    if (!stream ||
        !read_scalar_le(stream, num_features) ||
        !read_scalar_le(stream, ft_size) ||
        !read_scalar_le(stream, hidden_size) ||
        !read_scalar_le(stream, output_buckets) ||
        !read_scalar_le(stream, output_perspective) ||
        !read_scalar_le(stream, ft_scale) ||
        !read_scalar_le(stream, dense_scale) ||
        !read_scalar_le(stream, wdl_scale) ||
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
    if (ft_scale != static_cast<float>(kExpectedFtScale) ||
        dense_scale != static_cast<float>(kExpectedDenseScale)) {
        error = "unexpected quantization scales";
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
    std::vector<int32_t> out_bias_q;
    std::vector<int16_t> out_weight_q;

    if (!read_vector_le(stream, ft_bias_q, kExpectedFtSize) ||
        !read_vector_le(stream, ft_weight_q, static_cast<std::size_t>(kExpectedNumFeatures) * kExpectedFtSize) ||
        !read_vector_le(stream, l1_bias_q, kExpectedHiddenSize) ||
        !read_vector_le(stream, l1_weight_q, static_cast<std::size_t>(kExpectedFtSize) * 2 * kExpectedHiddenSize) ||
        !read_vector_le(stream, out_bias_q, kExpectedOutputBuckets) ||
        !read_vector_le(stream, out_weight_q, static_cast<std::size_t>(kExpectedHiddenSize) * kExpectedOutputBuckets)) {
        error = "failed while reading tensors";
        return false;
    }

    if (!stream) {
        error = "unexpected end of file";
        return false;
    }

    network = LoadedNetwork{};
    network.version = version;
    network.output_buckets = output_buckets;
    std::copy(ft_bias_q.begin(), ft_bias_q.end(), network.ft_bias.begin());
    network.ft_weight = std::move(ft_weight_q);
    std::copy(l1_bias_q.begin(), l1_bias_q.end(), network.l1_bias.begin());
    network.l1_weight = std::move(l1_weight_q);
    std::copy(out_bias_q.begin(), out_bias_q.end(), network.out_bias.begin());
    network.out_weight = std::move(out_weight_q);
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

    thrawn::NnueState refreshed_state;
    const thrawn::NnueState* state = state_for_evaluation(pos, *network, refreshed_state);
    if (state == nullptr) {
        return 0;
    }

    return static_cast<int>(
        (static_cast<int64_t>(evaluate_state_raw_qb2(*state, *network, pos->colour_to_move)) * 100) /
        kCpScaleDenominator
    );
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

    return static_cast<float>(evaluate_state_raw_qb2(*state, *network, pos->colour_to_move)) / kRawScale;
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
