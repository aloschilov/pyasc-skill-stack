# asc.language.basic.print_time_stamp

### asc.language.basic.print_time_stamp(desc_id: int) → None

提供时间戳打点功能，用于在算子Kernel代码中标记关键执行点。调用后会打印如下信息：

- desc_id： 用户自定义标识符，用于区分不同打点位置；
- rsv ：保留值，默认为0，无需关注；
- time_stamp ： 当前系统cycle数，用于计算时间差，时间换算规则可参考GetSystemCycle(ISASI)；
- pc_ptr：pc指针数值，若无特殊需求，用户无需关注。
- entry：预留字段，用户无需关注。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void PrintTimeStamp(uint32_t descId)
```
**参数说明**

- desc_id：用户自定义标识符（自定义数字），用于区分不同打点位置。[0, 0xffff]是预留给Ascend C内部各个模块使用的id值，用户自定义的desc_id建议使用大于0xffff的数值。

**约束说明**

- 该功能仅用于NPU上板调试。
- 暂不支持算子入图场景的打印。
- 单次调用本接口打印的数据总量不可超过1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。在使用自定义算子工程进行工程化算子开发时，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1MB。请开发者自行控制待打印的内容数据量，超出则不会打印。

**调用示例**

```python
asc.print_time_stamp(65577)
```
