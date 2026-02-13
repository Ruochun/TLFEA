#pragma once

#include <cuda_runtime.h>
// #include <cudss.h>
#include <cusparse.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

// Include MoPhiEssentials for GPU error handling and logging
#include <MoPhiEssentials.h>

// Use MoPhiEssentials' MOPHI_GPU_CALL macro for error handling
// For backward compatibility, alias HANDLE_ERROR to MOPHI_GPU_CALL
#ifndef HANDLE_ERROR
    #define HANDLE_ERROR(err) MOPHI_GPU_CALL(err)
#endif

#ifndef CHECK_CUSPARSE_MACRO
    #define CHECK_CUSPARSE_MACRO
    #define CHECK_CUSPARSE(func)                                                         \
        {                                                                                \
            cusparseStatus_t status = (func);                                            \
            if (status != CUSPARSE_STATUS_SUCCESS) {                                     \
                MOPHI_ERROR("CUSPARSE API failed with error: %s (%d)",                   \
                       cusparseGetErrorString(status), status);                          \
            }                                                                            \
        }
#endif

#ifndef CHECK_CUDSS_MACRO
    #define CHECK_CUDSS_MACRO
    #define CUDSS_OK(call)                                                                  \
        do {                                                                                \
            cudssStatus_t status = call;                                                    \
            if (status != CUDSS_STATUS_SUCCESS) {                                           \
                MOPHI_ERROR("cuDSS error");                                                 \
            }                                                                               \
        } while (0)
#endif