# asc.language.fwk.TBuf._\_init_\_

#### TBuf.\_\_init_\_(pos: TPosition) → None

#### TBuf.\_\_init_\_(handle: Value) → None

创建TBuf对象时，初始化数据成员。

**对应的Ascend C函数原型**

```c++
template <TPosition pos = TPosition::LCM>
__aicore__ inline TBuf();
```

**参数说明**

- pos：TBuf所在的逻辑位置，取值为VECCALC。

**约束说明**

无。
