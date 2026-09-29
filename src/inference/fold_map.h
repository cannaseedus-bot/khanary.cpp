#pragma once
#include "../util/json.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace khanary::inference {

enum class TensorDtype { F32, Q8_0, MXFP4, F16, U8, Unknown };

// One contiguous slice of a tensor within a physical DDS shard file.
struct ShardSlice {
    std::string path;              // absolute path to fold_N.dds
    uint64_t    offset_in_file;    // bytes from file start (includes 128-byte DDS header)
    uint64_t    size;              // byte count of this slice
    uint64_t    offset_in_tensor;  // position within the full logical tensor
};

// One logical tensor — may span multiple ShardSlices across physical files.
struct TensorRef {
    std::string             name;
    TensorDtype             dtype;
    std::vector<uint32_t>   logical_shape;  // pre-quantization shape from kson
    std::vector<ShardSlice> shards;         // ordered by offset_in_tensor
};

// One fold = one logical layer group (embedding, layer_NN.attention, layer_NN.moe, output).
struct FoldEntry {
    uint32_t               fold_index;
    std::string            layer_id;  // e.g. "layer_03.attention", "embedding"
    float                  vector_x = 0.0f;
    float                  vector_y = 0.0f;
    float                  vector_z = 0.0f;
    bool                   has_vector = false;
    std::vector<TensorRef> tensors;
};

// Loaded from gpt_oss_20b_folds.kson + folds/model_dds_manifest.json.
// Maps layer_id / fold_index → tensors → physical ShardSlices.
// Primary source of truth for InferenceEngine DDS hot-swap.
class FoldMap {
public:
    // kson_path    : path to gpt_oss_20b_folds.kson
    // manifest_path: path to folds/model_dds_manifest.json
    // shards_dir   : directory containing fold_0.dds … fold_90.dds
    static FoldMap load(const std::string& kson_path,
                        const std::string& manifest_path,
                        const std::string& shards_dir);

    const FoldEntry*  get(const std::string& layer_id) const;
    const FoldEntry*  get(uint32_t fold_index) const;
    const TensorRef*  get_tensor(const std::string& tensor_name) const;

    size_t fold_count() const { return folds_.size(); }
    const std::vector<FoldEntry>& folds() const { return folds_; }
    bool empty() const { return folds_.empty(); }

    static TensorDtype  parse_dtype(const std::string& s);
    static std::string  dtype_name(TensorDtype d);

private:
    std::vector<FoldEntry>                          folds_;
    std::unordered_map<std::string, size_t>         by_layer_id_;
    std::unordered_map<uint32_t,   size_t>          by_fold_index_;
    std::unordered_map<std::string, const TensorRef*> by_tensor_name_;

    void build_indices();
};

} // namespace khanary::inference
