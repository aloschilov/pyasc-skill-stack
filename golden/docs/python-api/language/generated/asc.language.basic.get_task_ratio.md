# asc.language.basic.get_task_ratio

### asc.language.basic.get_task_ratio() → int

分离模式下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量与AI Core数量的比例。耦合模式下，固定返回1。

**对应的 Ascend C 函数原型**

```c++
__aicore__ inline int64_t GetTaskRatio();
```

**参数说明**

无。

**返回值说明**

针对分离模式，不同Kernel类型下（通过设置Kernel类型设置），在AIC和AIV上调用该接口的返回值如下：

表1 返回值列表

| Kernel 类型 | KERNEL_TYPE_AIV_ONLY | KERNEL_TYPE_AIC_ONLY | KERNEL_TYPE_MIX_AIC_1_2 | KERNEL_TYPE_MIX_AIC_1_1 | KERNEL_TYPE_MIX_AIC_1_0 | KERNEL_TYPE_MIX_AIV_1_0 |
| ----------- | -------------------- | -------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| AIV         | 1                    | －                   | 2                       | 1                       | －                      | 1                       |
| AIC         | －                   | 1                    | 1                       | 1                       | 1                       | －                      |

针对耦合模式，固定返回 1。

**约束说明**

无。

**调用示例**

```python
import asc
ratio = asc.get_task_ratio()
```
