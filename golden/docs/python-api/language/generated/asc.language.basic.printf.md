# asc.language.basic.printf

### asc.language.basic.printf(desc: str, \*params) → None

该接口提供CPU域/NPU域调试场景下的格式化输出功能。
在算子kernel侧实现代码中需要输出日志信息的地方调用printf接口打印相关内容。

**对应的Ascend C函数原型**

```c++
void printf(__gm__ const char* fmt, Args&&... args)
void PRINTF(__gm__ const char* fmt, Args&&... args)
```

**参数说明**

- fmt：格式控制字符串，包含两种类型的对象：普通字符和转换说明。
- args：附加参数，个数和类型可变的参数列表。

**约束说明**

- 本接口不支持打印除换行符之外的其他转义字符。
- 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
- 算子入图场景，若一个动态Shape模型中有可下沉的部分，框架内部会将模型拆分为动态调度和下沉调度（静态子图）两部分，静态子图中的算子不支持该printf特性。

**调用示例**

```python
#整型打印
x = 10
asc.printf("%d", x)
#浮点型打印
x = 3.14
asc.printf("%f", x)
```
