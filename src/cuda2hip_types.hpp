#pragma once
// Type mappings for CUDA to HIP compatibility (safe for C++ compilation)
#ifdef __HIP_PLATFORM_AMD__
  // cuBLAS types
  #define cublasStatus_t hipblasStatus_t
  #define cublasHandle_t hipblasHandle_t

  // CUDA runtime types
  #define cudaStream_t hipStream_t
  #define cudaError_t hipError_t
  #define cudaEvent_t hipEvent_t
  #define cudaDeviceProp hipDeviceProp_t

  // cuDNN types (MIOpen equivalents)
  #ifdef CT2_WITH_CUDNN
    #define cudnnHandle_t miopenHandle_t
    #define cudnnDataType_t miopenDataType_t
  #endif
#endif
