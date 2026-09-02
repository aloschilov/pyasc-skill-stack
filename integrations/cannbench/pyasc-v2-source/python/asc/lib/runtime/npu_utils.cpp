/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/syscall.h>

#include "acl/acl.h"
#ifdef SEPARATE_PKG_ARCH
#include "profiling/aprof_pub.h"
#include "profiling/prof_api.h"
#include "profiling/prof_common.h"
#include "runtime/rt.h"
#else
#include "experiment/msprof/toolchain/prof_api.h"
#include "experiment/msprof/toolchain/prof_data_config.h"
#include "experiment/runtime/runtime/rt.h"
#endif

static unsigned int moduleId = 8;
static unsigned int msprofFlagL0 = 0;
static unsigned int msprofFlagL1 = 0;

extern "C" {
int ProfCtrlHandle(unsigned int ctrlType, void* ctrlData, unsigned int dataLen)
{
    if (ctrlType != PROF_CTRL_SWITCH || ctrlData == nullptr || dataLen < sizeof(MsprofCommandHandle)) {
        return 1;
    }

    MsprofCommandHandle* handle = static_cast<MsprofCommandHandle*>(ctrlData);
    const uint64_t profSwitch = handle->profSwitch;
    const uint64_t profType = handle->type;
    if (profType == PROF_COMMANDHANDLE_TYPE_START) {
#ifdef SEPARATE_PKG_ARCH
        if ((profSwitch & PROF_TASK_TIME_MASK) == PROF_TASK_TIME_MASK) {
#else
        if ((profSwitch & PROF_TASK_TIME) == PROF_TASK_TIME) {
#endif
            msprofFlagL0 = 1;
        }
#ifdef SEPARATE_PKG_ARCH
        if ((profSwitch & PROF_TASK_TIME_L1_MASK) == PROF_TASK_TIME_L1_MASK) {
#else
        if ((profSwitch & PROF_TASK_TIME_L1) == PROF_TASK_TIME_L1) {
#endif
            msprofFlagL1 = 1;
        }
    }
    if (profType == PROF_COMMANDHANDLE_TYPE_STOP) {
        msprofFlagL0 = 0;
        msprofFlagL1 = 0;
    }
    return 0;
}
}

static PyObject* aclInit(PyObject* self, PyObject* args)
{
    aclError ret = aclInit(nullptr);
    if (PyErr_Occurred()) {
        return nullptr;
    }

    return Py_BuildValue("i", ret);
}

static PyObject* aclFinalize(PyObject* self, PyObject* args)
{
    aclError ret = aclFinalize();
    if (PyErr_Occurred()) {
        return nullptr;
    }

    return Py_BuildValue("i", ret);
}

static PyObject* MsprofSysCycleTime(PyObject* self, PyObject* args)
{
    if (!msprofFlagL0 && !msprofFlagL1) {
        return Py_BuildValue("k", 0);
    }

    uint64_t time = MsprofSysCycleTime();

    if (PyErr_Occurred()) {
        return nullptr;
    }

    return Py_BuildValue("k", time);
}

static PyObject* MsprofReportApi(PyObject* self, PyObject* args)
{
    if (!msprofFlagL0 && !msprofFlagL1) {
        return Py_BuildValue("i", 1);
    }

    unsigned long start = 0;
    unsigned long end = 0;
    const char* opName = "";

    if (!PyArg_ParseTuple(args, "kks", &start, &end, &opName)) {
        return nullptr;
    }

    long threadId = syscall(SYS_gettid);
    uint64_t hashId = MsprofGetHashId(opName, strlen(opName));
    MsprofApi api;
    api.level = MSPROF_REPORT_NODE_LEVEL;
    api.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    api.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    api.threadId = threadId;
    api.reserve = 0;
    api.beginTime = start;
    api.endTime = end;
    api.itemId = hashId;
    int32_t ret = MsprofReportApi(false, &api);

    if (PyErr_Occurred()) {
        return nullptr;
    }

    return Py_BuildValue("i", ret);
}

static PyObject* MsprofReportCompactInfo(PyObject* self, PyObject* args)
{
    if (!msprofFlagL1) {
        return Py_BuildValue("i", 1);
    }

    unsigned long time;
    const char* opName;
    unsigned int blockNum;
    unsigned int taskType;

    if (!PyArg_ParseTuple(args, "ksII", &time, &opName, &blockNum, &taskType)) {
        return nullptr;
    }
    uint64_t hashId = MsprofGetHashId(opName, strlen(opName));
    long threadId = syscall(SYS_gettid);
    MsprofCompactInfo nodeBasicInfo;
    nodeBasicInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM; // MSPROF_REPORT_DATA_MAGIC_NUM
    nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;           // MSPROF_REPORT_NODE_LEVEL
    nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;  // MSPROF_REPORT_NODE_BASIC_INFO_TYPE
    nodeBasicInfo.threadId = threadId;
    nodeBasicInfo.timeStamp = time;
    nodeBasicInfo.data.nodeBasicInfo.opName = hashId;
    nodeBasicInfo.data.nodeBasicInfo.taskType = taskType; // MSPROF_GE_TASK_TYPE_AI_CORE
    nodeBasicInfo.data.nodeBasicInfo.opType = hashId;
    nodeBasicInfo.data.nodeBasicInfo.blockDim = blockNum;
    int32_t ret = MsprofReportCompactInfo(0, &nodeBasicInfo, sizeof(MsprofCompactInfo));

    if (PyErr_Occurred()) {
        return nullptr;
    }

    return Py_BuildValue("i", ret);
}

static PyObject* MsprofReportAdditionalInfo(PyObject* self, PyObject* args)
{
    if (!msprofFlagL1) {
        return Py_BuildValue("i", 1);
    }
    unsigned long time;
    const char* opName;
    if (!PyArg_ParseTuple(args, "ks", &time, &opName)) {
        return nullptr;
    }

    uint64_t hashId = MsprofGetHashId(opName, strlen(opName));
    long threadId = syscall(SYS_gettid);

    MsprofAdditionalInfo tensorInfo;
    tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
    tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
    tensorInfo.threadId = threadId;
    tensorInfo.timeStamp = time;
    auto profTensorData = reinterpret_cast<MsprofTensorInfo*>(tensorInfo.data);
    profTensorData->opName = hashId;

    int32_t ret = MsprofReportAdditionalInfo(false, static_cast<void*>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    if (PyErr_Occurred()) {
        return nullptr;
    }

    return Py_BuildValue("i", ret);
}

static PyMethodDef NpuUtilsMethods[] = {
    {"acl_init", aclInit, METH_NOARGS, "aclInit"},
    {"acl_finalize", aclFinalize, METH_NOARGS, "aclFinalize"},
    {"msprof_sys_cycle_time", MsprofSysCycleTime, METH_VARARGS, "MsprofSysCycleTime"},
    {"msprof_report_api", MsprofReportApi, METH_VARARGS, "MsprofReportApi"},
    {"msprof_report_compact_info", MsprofReportCompactInfo, METH_VARARGS, "MsprofReportCompactInfo"},
    {"msprof_report_additional_info", MsprofReportAdditionalInfo, METH_VARARGS, "MsprofReportAdditionalInfo"},
    {nullptr, nullptr, 0, nullptr}};

static PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "npu_utils", "Npu utils", -1, NpuUtilsMethods};

PyMODINIT_FUNC PyInit_npu_utils(void)
{
    PyObject* m = PyModule_Create(&ModuleDef);
    if (m == nullptr) {
        return nullptr;
    }

    PyModule_AddFunctions(m, NpuUtilsMethods);
    aclInit(nullptr);
    MsprofRegisterCallback(moduleId, ProfCtrlHandle);
    return m;
}
