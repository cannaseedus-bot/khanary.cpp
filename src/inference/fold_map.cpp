#include "fold_map.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace khanary::inference {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("FoldMap: cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
    return dir + '/' + file;
}

// ---------------------------------------------------------------------------
// dtype
// ---------------------------------------------------------------------------

TensorDtype FoldMap::parse_dtype(const std::string& s) {
    // kson uses lowercase; manifest uses uppercase
    if (s == "f32"  || s == "F32")  return TensorDtype::F32;
    if (s == "f16"  || s == "F16")  return TensorDtype::F16;
    if (s == "q8_0" || s == "Q8_0") return TensorDtype::Q8_0;
    if (s == "mxfp4"|| s == "MXFP4")return TensorDtype::MXFP4;
    if (s == "u8"   || s == "U8")   return TensorDtype::U8;
    return TensorDtype::Unknown;
}

std::string FoldMap::dtype_name(TensorDtype d) {
    switch (d) {
        case TensorDtype::F32:    return "f32";
        case TensorDtype::F16:    return "f16";
        case TensorDtype::Q8_0:   return "q8_0";
        case TensorDtype::MXFP4:  return "mxfp4";
        case TensorDtype::U8:     return "u8";
        default:                  return "unknown";
    }
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

FoldMap FoldMap::load(const std::string& kson_path,
                      const std::string& manifest_path,
                      const std::string& shards_dir) {

    // ---- 1. Parse model_dds_manifest.json ---------------------------------
    // Format: JSON array of 91 shard objects.
    //   { "id": N, "dds_file": "fold_N.dds", "tensors": [
    //       { "name": "...", "offset_in_file": N, "offset_in_tensor": N,
    //         "size": N, "dtype": "U8", "shape": [...], "logical_shape": [...] }
    //   ]}
    //
    // Build: tensor_name → vector<ShardSlice> (sorted by offset_in_tensor)

    std::unordered_map<std::string, std::vector<ShardSlice>> tensor_shards;
    std::unordered_map<std::string, TensorDtype>             tensor_dtype;
    std::unordered_map<std::string, std::vector<uint32_t>>   tensor_logical_shape;

    {
        auto manifest_root = util::parse(read_file(manifest_path));
        if (!manifest_root || !manifest_root->is_array())
            throw std::runtime_error("FoldMap: manifest must be a JSON array");

        for (const auto& shard_ptr : manifest_root->as_array()) {
            if (!shard_ptr || !shard_ptr->is_object()) continue;
            const auto& shard_obj = shard_ptr->as_object();

            auto it_file = shard_obj.find("dds_file");
            if (it_file == shard_obj.end()) continue;
            std::string dds_file = it_file->second->as_string();
            std::string full_path = join_path(shards_dir, dds_file);

            auto it_tensors = shard_obj.find("tensors");
            if (it_tensors == shard_obj.end()) continue;
            if (!it_tensors->second->is_array()) continue;

            for (const auto& t_ptr : it_tensors->second->as_array()) {
                if (!t_ptr || !t_ptr->is_object()) continue;
                const auto& t = t_ptr->as_object();

                auto it_name = t.find("name");
                if (it_name == t.end()) continue;
                std::string tname = it_name->second->as_string();

                ShardSlice slice;
                slice.path = full_path;

                auto it_off = t.find("offset_in_file");
                slice.offset_in_file = (it_off != t.end())
                    ? static_cast<uint64_t>(it_off->second->as_number()) : 0;

                auto it_sz = t.find("size");
                slice.size = (it_sz != t.end())
                    ? static_cast<uint64_t>(it_sz->second->as_number()) : 0;

                auto it_ot = t.find("offset_in_tensor");
                slice.offset_in_tensor = (it_ot != t.end())
                    ? static_cast<uint64_t>(it_ot->second->as_number()) : 0;

                tensor_shards[tname].push_back(slice);

                // dtype from manifest (authoritative for physical encoding)
                if (tensor_dtype.find(tname) == tensor_dtype.end()) {
                    auto it_dt = t.find("dtype");
                    if (it_dt != t.end())
                        tensor_dtype[tname] = parse_dtype(it_dt->second->as_string());

                    auto it_ls = t.find("logical_shape");
                    if (it_ls != t.end() && it_ls->second->is_array()) {
                        for (const auto& d : it_ls->second->as_array())
                            tensor_logical_shape[tname].push_back(
                                static_cast<uint32_t>(d->as_number()));
                    }
                }
            }
        }

        // Sort each tensor's slices by offset_in_tensor
        for (auto& [name, slices] : tensor_shards) {
            std::sort(slices.begin(), slices.end(),
                [](const ShardSlice& a, const ShardSlice& b) {
                    return a.offset_in_tensor < b.offset_in_tensor;
                });
        }
    }

    // ---- 2. Parse gpt_oss_20b_folds.kson ----------------------------------
    // Format: kast/1 object with @driver.@resources.folds array.
    //   { "fold": N, "id": "layer_00.attention", "shard": "FOLD_001.dds",
    //     "nodes": [ { "name": "blk.0.attn_k.bias", "shape": [...],
    //                  "scxq2": "q8_0", "offset": N, "length": N } ] }

    FoldMap fm;

    {
        auto kson_root = util::parse(read_file(kson_path));
        if (!kson_root || !kson_root->is_object())
            throw std::runtime_error("FoldMap: kson must be a JSON object");

        const util::JsonValue* driver = kson_root->get("@driver");
        if (!driver)
            throw std::runtime_error("FoldMap: missing @driver in kson");

        const util::JsonValue* resources = driver->get("@resources");
        if (!resources)
            throw std::runtime_error("FoldMap: missing @driver.@resources in kson");

        const util::JsonValue* folds_val = resources->get("folds");
        if (!folds_val || !folds_val->is_array())
            throw std::runtime_error("FoldMap: missing @driver.@resources.folds array");

        for (const auto& fold_ptr : folds_val->as_array()) {
            if (!fold_ptr || !fold_ptr->is_object()) continue;
            const auto& fold_obj = fold_ptr->as_object();

            FoldEntry entry;

            auto it_idx = fold_obj.find("fold");
            entry.fold_index = (it_idx != fold_obj.end())
                ? static_cast<uint32_t>(it_idx->second->as_number()) : 0;

            auto it_id = fold_obj.find("id");
            entry.layer_id = (it_id != fold_obj.end())
                ? it_id->second->as_string() : ("fold_" + std::to_string(entry.fold_index));

            auto parse_vector_object = [&](const util::JsonValue* vec_obj) {
                if (!vec_obj || !vec_obj->is_object()) return;
                const auto vx = vec_obj->get_number("x");
                const auto vy = vec_obj->get_number("y");
                const auto vz = vec_obj->get_number("z");
                if (!vx || !vy || !vz) return;
                if (!std::isfinite(*vx) || !std::isfinite(*vy) || !std::isfinite(*vz)) return;
                entry.vector_x = static_cast<float>(*vx);
                entry.vector_y = static_cast<float>(*vy);
                entry.vector_z = static_cast<float>(*vz);
                entry.has_vector = true;
            };
            if (auto it_vec = fold_obj.find("vector"); it_vec != fold_obj.end())
                parse_vector_object(it_vec->second.get());
            if (!entry.has_vector) {
                if (auto it_vec = fold_obj.find("qkv_vector"); it_vec != fold_obj.end())
                    parse_vector_object(it_vec->second.get());
            }

            auto it_nodes = fold_obj.find("nodes");
            if (it_nodes != fold_obj.end() && it_nodes->second->is_array()) {
                for (const auto& node_ptr : it_nodes->second->as_array()) {
                    if (!node_ptr || !node_ptr->is_object()) continue;
                    const auto& node = node_ptr->as_object();

                    auto it_name = node.find("name");
                    if (it_name == node.end()) continue;
                    std::string tname = it_name->second->as_string();

                    TensorRef tref;
                    tref.name = tname;

                    // dtype: prefer kson scxq2 field, fall back to manifest
                    auto it_scxq = node.find("scxq2");
                    if (it_scxq != node.end())
                        tref.dtype = parse_dtype(it_scxq->second->as_string());
                    else if (tensor_dtype.count(tname))
                        tref.dtype = tensor_dtype.at(tname);
                    else
                        tref.dtype = TensorDtype::Unknown;

                    // logical_shape from kson node
                    auto it_shape = node.find("shape");
                    if (it_shape != node.end() && it_shape->second->is_array()) {
                        for (const auto& d : it_shape->second->as_array())
                            tref.logical_shape.push_back(
                                static_cast<uint32_t>(d->as_number()));
                    } else if (tensor_logical_shape.count(tname)) {
                        tref.logical_shape = tensor_logical_shape.at(tname);
                    }

                    // physical shards from manifest
                    auto it_sl = tensor_shards.find(tname);
                    if (it_sl != tensor_shards.end())
                        tref.shards = it_sl->second;

                    entry.tensors.push_back(std::move(tref));
                }
            }

            fm.folds_.push_back(std::move(entry));
        }
    }

    fm.build_indices();
    return fm;
}

// ---------------------------------------------------------------------------
// index building
// ---------------------------------------------------------------------------

void FoldMap::build_indices() {
    by_layer_id_.clear();
    by_fold_index_.clear();
    by_tensor_name_.clear();

    for (size_t i = 0; i < folds_.size(); ++i) {
        by_layer_id_[folds_[i].layer_id]      = i;
        by_fold_index_[folds_[i].fold_index]   = i;
        for (const auto& t : folds_[i].tensors)
            by_tensor_name_[t.name] = &t;
    }
}

// ---------------------------------------------------------------------------
// accessors
// ---------------------------------------------------------------------------

const FoldEntry* FoldMap::get(const std::string& layer_id) const {
    auto it = by_layer_id_.find(layer_id);
    return (it != by_layer_id_.end()) ? &folds_[it->second] : nullptr;
}

const FoldEntry* FoldMap::get(uint32_t fold_index) const {
    auto it = by_fold_index_.find(fold_index);
    return (it != by_fold_index_.end()) ? &folds_[it->second] : nullptr;
}

const TensorRef* FoldMap::get_tensor(const std::string& tensor_name) const {
    auto it = by_tensor_name_.find(tensor_name);
    return (it != by_tensor_name_.end()) ? it->second : nullptr;
}

} // namespace khanary::inference
