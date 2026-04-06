<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# asc.runtime.config

Ascend C运行时配置接口，用户通过该模块可配置后端执行模式、SOC版本和设备ID等信息。

## 接口列表

| `set_platform`(backend[, soc_version, device_id, check])                            | 设置运行时后端、SOC版本和设备ID。当backend为Model时，soc_version默认为Ascend910B1；当backend为NPU时，soc_version自动从当前平台获取，且会校验与输入的soc_version是否一致。|
|-------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|

## 枚举类型

### Backend

指定后端执行模式。

| 枚举值 | 说明 |
|--------|------|
| Backend.Model | 使用Model后端执行，适用于仿真或模型运行场景 |
| Backend.NPU | 使用NPU后端执行，适用于真实NPU硬件场景 |

### Platform

指定SOC版本。

| 枚举值 | 说明 |
|--------|------|
| Platform.Ascend910B1 | Ascend 910B1 |
| Platform.Ascend910B2 | Ascend 910B2 |
| Platform.Ascend910B2C | Ascend 910B2C |
| Platform.Ascend910B3 | Ascend 910B3 |
| Platform.Ascend910B4 | Ascend 910B4 |
| Platform.Ascend910B4_1 | Ascend 910B4-1 |
| Platform.Ascend910_9362 | Ascend 910 9362 |
| Platform.Ascend910_9372 | Ascend 910 9372 |
| Platform.Ascend910_9381 | Ascend 910 9381 |
| Platform.Ascend910_9382 | Ascend 910 9382 |
| Platform.Ascend910_9391 | Ascend 910 9391 |
| Platform.Ascend910_9392 | Ascend 910 9392 |