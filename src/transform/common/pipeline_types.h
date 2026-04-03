// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TL_TRANSFORM_COMMON_PIPELINE_TYPES_H_
#define TL_TRANSFORM_COMMON_PIPELINE_TYPES_H_

#include <string>

#include <tvm/runtime/logging.h>

namespace tvm {
namespace tl {

enum class PipelineType : int {
  PIPE_V = 0,
  PIPE_M = 1,
  PIPE_MTE1 = 2,
  PIPE_MTE2 = 3,
  PIPE_MTE3 = 4,
  PIPE_FIX = 5,
  PIPE_S = 6,
  PIPE_ALL = 7,
};

constexpr int kPipelineUnset = -1;

inline std::string PipelineTypeToString(int pipeline_int) {
  if (pipeline_int == kPipelineUnset) {
    return "";
  }

  switch (static_cast<PipelineType>(pipeline_int)) {
    case PipelineType::PIPE_V:
      return "PIPE_V";
    case PipelineType::PIPE_M:
      return "PIPE_M";
    case PipelineType::PIPE_MTE1:
      return "PIPE_MTE1";
    case PipelineType::PIPE_MTE2:
      return "PIPE_MTE2";
    case PipelineType::PIPE_MTE3:
      return "PIPE_MTE3";
    case PipelineType::PIPE_FIX:
      return "PIPE_FIX";
    case PipelineType::PIPE_S:
      return "PIPE_S";
    case PipelineType::PIPE_ALL:
      return "PIPE_ALL";
  }

  LOG(FATAL) << "Invalid pipeline type value: " << pipeline_int;
  return "";
}

inline int StringToPipelineType(const std::string& name) {
  if (name == "PIPE_V") {
    return static_cast<int>(PipelineType::PIPE_V);
  }
  if (name == "PIPE_M") {
    return static_cast<int>(PipelineType::PIPE_M);
  }
  if (name == "PIPE_MTE1") {
    return static_cast<int>(PipelineType::PIPE_MTE1);
  }
  if (name == "PIPE_MTE2") {
    return static_cast<int>(PipelineType::PIPE_MTE2);
  }
  if (name == "PIPE_MTE3") {
    return static_cast<int>(PipelineType::PIPE_MTE3);
  }
  if (name == "PIPE_FIX") {
    return static_cast<int>(PipelineType::PIPE_FIX);
  }
  if (name == "PIPE_S") {
    return static_cast<int>(PipelineType::PIPE_S);
  }
  if (name == "PIPE_ALL") {
    return static_cast<int>(PipelineType::PIPE_ALL);
  }
  return kPipelineUnset;
}

inline bool IsValidPipelineValue(int val) { return val >= kPipelineUnset && val <= 7; }

}  // namespace tl
}  // namespace tvm

#endif  // TL_TRANSFORM_COMMON_PIPELINE_TYPES_H_
