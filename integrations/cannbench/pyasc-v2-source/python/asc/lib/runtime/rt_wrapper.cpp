/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef SEPARATE_PKG_ARCH
#include "runtime/rt.h"
#else
#include "experiment/runtime/runtime/rt.h"
#endif

extern "C" {
/**
 * @ingroup dvrt_dev
 * @brief get chipType
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t GetSocVersionWrapper(char_t* ver, const uint32_t maxLen) { return rtGetSocVersion(ver, maxLen); }

/**
 * @ingroup dvrt_dev
 * @brief get total device number.
 * @param [in|out] cnt the device number
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t GetDeviceCountWrapper(int32_t* cnt) { return rtGetDeviceCount(cnt); }

/**
 * @ingroup dvrt_dev
 * @brief get device infomation.
 * @param [in] device   the device id
 * @param [in] moduleType   module type
               typedef enum {
                    MODULE_TYPE_SYSTEM = 0,   system info
                    MODULE_TYPE_AICPU,        aicpu info
                    MODULE_TYPE_CCPU,         ccpu_info
                    MODULE_TYPE_DCPU,         dcpu info
                    MODULE_TYPE_AICORE,       AI CORE info
                    MODULE_TYPE_TSCPU,        tscpu info
                    MODULE_TYPE_PCIE,         PCIE info
               } DEV_MODULE_TYPE;
 * @param [in] infoType   info type
               typedef enum {
                    INFO_TYPE_ENV = 0,
                    INFO_TYPE_VERSION,
                    INFO_TYPE_MASTERID,
                    INFO_TYPE_CORE_NUM,
                    INFO_TYPE_OS_SCHED,
                    INFO_TYPE_IN_USED,
                    INFO_TYPE_ERROR_MAP,
                    INFO_TYPE_OCCUPY,
                    INFO_TYPE_ID,
                    INFO_TYPE_IP,
                    INFO_TYPE_ENDIAN,
               } DEV_INFO_TYPE;
 * @param [out] val   the device info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_DRV_ERR for error
 */
RTS_API rtError_t GetDeviceInfoWrapper(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t* val)
{
    return rtGetDeviceInfo(deviceId, moduleType, infoType, val);
}

/**
 * @ingroup dvrt_dev
 * @brief reset all opened device
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t DeviceResetWrapper(int32_t devId) { return rtDeviceReset(devId); }

/**
 * @ingroup dvrt_dev
 * @brief set target device for current thread
 * @param [int] devId   the device id
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t SetDeviceWrapper(int32_t devId) { return rtSetDevice(devId); }

/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stm   created stream
 * @param [in] priority   stream priority
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t StreamCreateWrapper(rtStream_t* stm, int32_t priority) { return rtStreamCreate(stm, priority); }

/**
 * @ingroup dvrt_stream
 * @brief destroy stream instance.
 * @param [in] stm   the stream to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t StreamDestroyWrapper(rtStream_t stm) { return rtStreamDestroy(stm); }

/**
 * @ingroup rt_kernel
 * @brief register device binary
 * @param [in] bin   device binary description
 * @param [out] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t DevBinaryRegisterWrapper(const rtDevBinary_t* bin, void** hdl)
{
    return rtDevBinaryRegister(bin, hdl);
}

/**
 * @ingroup rt_kernel
 * @brief unregister device binary
 * @param [in] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t DevBinaryUnRegisterWrapper(void* hdl) { return rtDevBinaryUnRegister(hdl); }

/**
 * @ingroup rt_kernel
 * @brief register device function
 * @param [in] binHandle   device binary handle
 * @param [in] stubFunc   stub function
 * @param [in] stubName   stub function name
 * @param [in] kernelInfoExt   kernel Info extension. device function description or tiling key,
 *                             depending static shape or dynmaic shape.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t FunctionRegisterWrapper(
    void* binHandle, const void* stubFunc, const char_t* stubName, const void* kernelInfoExt, uint32_t funcMode)
{
    return rtFunctionRegister(binHandle, stubFunc, stubName, kernelInfoExt, funcMode);
}

/**
 * @ingroup dvrt_mem
 * @brief alloc device memory
 * @param [in|out] devPtr   memory pointer
 * @param [in] size   memory size
 * @param [in] type   memory type
 * @param [in] moduleid alloc memory module id
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t MallocWrapper(void** devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    return rtMalloc(devPtr, size, type, moduleId);
}

/**
 * @ingroup dvrt_mem
 * @brief synchronized memcpy
 * @param [in] dst     destination address pointer
 * @param [in] destMax length of destination address memory
 * @param [in] src   source address pointer
 * @param [in] cnt   the number of byte to copy
 * @param [in] kind  memcpy type
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t MemcpyWrapper(void* dst, uint64_t destMax, const void* src, uint64_t cnt, rtMemcpyKind_t kind)
{
    return rtMemcpy(dst, destMax, src, cnt, kind);
}

/**
 * @ingroup rt_kernel
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] blockNum   block dimentions
 * @param [in] args   argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] smDesc   shared memory description
 * @param [in] stm   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t KernelLaunchWrapper(
    const void* stubFunc, uint32_t blockNum, void* args, uint32_t argsSize, rtSmDesc_t* smDesc, rtStream_t stm)
{
    return rtKernelLaunch(stubFunc, blockNum, args, argsSize, smDesc, stm);
}

/**
 * @ingroup dvrt_stream
 * @brief wait stream to be complete and set timeout
 * @param [in] stm   stream to wait
 * @param [in] timeout   timeout value,the unit is milliseconds
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t StreamSynchronizeWithTimeoutWrapper(rtStream_t stm, int32_t timeout)
{
    return rtStreamSynchronizeWithTimeout(stm, timeout);
}

/**
 * @ingroup dvrt_stream
 * @brief wait stream to be complete
 * @param [in] stm   stream to wait
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t StreamSynchronizeWrapper(rtStream_t stm) { return rtStreamSynchronize(stm); }

/**
 * @ingroup dvrt_mem
 * @brief free device memory
 * @param [in|out] devPtr   memory pointer
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t FreeWrapper(void* devPtr) { return rtFree(devPtr); }

/**
 * @ingroup dvrt_dev
 * @brief Wait for compute device to finish
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t DeviceSynchronizeWrapper(void) { return rtDeviceSynchronize(); }

RTS_API rtError_t GetC2cCtrlAddrWrapper(uint64_t* addr, uint32_t* len) { return rtGetC2cCtrlAddr(addr, len); }

/**
 * @ingroup profiling_base
 * @brief set profling switch, called by profiling
 * @param [in]  data  rtProfCommandHandle
 * @param [out] len   length of data
 * @return RT_ERROR_NONE for ok
 * @return ACL_ERROR_RT_PARAM_INVALID for error input
 */
RTS_API rtError_t ProfSetProSwitchWrapper(void* data, uint32_t len) { return rtProfSetProSwitch(data, len); }
}
