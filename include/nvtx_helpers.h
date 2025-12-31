//
// Created by Owner on 15/12/2025.
//

#pragma once

#ifdef ENABLE_NVTX
  #include <nvtx3/nvToolsExt.h>
  #define NVTX_MARK(msg) nvtxMarkA(msg)

  struct NvtxRange {
      explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
      ~NvtxRange() { nvtxRangePop(); }
  };

#define NVTX_CONCAT2(a,b) a##b
#define NVTX_CONCAT(a,b) NVTX_CONCAT2(a,b)
#define NVTX_RANGE(name) NvtxRange NVTX_CONCAT(_nvtx_, __LINE__)(name)
#else
#define NVTX_MARK(msg) ((void)0)
#define NVTX_RANGE(name) ((void)0)
#endif
