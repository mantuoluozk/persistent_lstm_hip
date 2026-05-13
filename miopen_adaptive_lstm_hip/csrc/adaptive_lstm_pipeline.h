#pragma once

namespace adaptive_lstm {

enum class PipelineVersion {
  kScalar,
  kPartitioned,
  kCachedB2,
  kXdlopsV1,
  kXdlopsV2,
};

template <
    PipelineVersion Version,
    int BlockSizeValue,
    int BatchTileValue,
    int HiddenSizeValue,
    int KPerBlockValue,
    int PartitionsValue,
    int NumKPrefetchStageValue>
struct RecurrentPipelineTraits {
  static constexpr PipelineVersion kVersion = Version;
  static constexpr int kBlockSize = BlockSizeValue;
  static constexpr int kBatchTile = BatchTileValue;
  static constexpr int kHiddenSize = HiddenSizeValue;
  static constexpr int kKPerBlock = KPerBlockValue;
  static constexpr int kPartitions = PartitionsValue;
  static constexpr int kNumKPrefetchStage = NumKPrefetchStageValue;
};

using H128CachedB2Traits = RecurrentPipelineTraits<
    PipelineVersion::kCachedB2,
    512,
    2,
    128,
    32,
    4,
    1>;

using H128CachedB4Traits = RecurrentPipelineTraits<
    PipelineVersion::kCachedB2,
    512,
    4,
    128,
    32,
    4,
    1>;

using H128CachedB8Traits = RecurrentPipelineTraits<
    PipelineVersion::kCachedB2,
    512,
    8,
    128,
    32,
    4,
    1>;

using GenericPartitionedTraits = RecurrentPipelineTraits<
    PipelineVersion::kPartitioned,
    256,
    1,
    -1,
    32,
    4,
    1>;

}  // namespace adaptive_lstm
